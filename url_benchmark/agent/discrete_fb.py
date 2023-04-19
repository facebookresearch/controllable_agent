# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pylint: disable=unused-import
import pdb
import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore

from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import BackwardMap, mlp


logger = logging.getLogger(__name__)


from .fb_ddpg import FBDDPGAgentConfig


@dataclasses.dataclass
class DiscreteFBAgentConfig(FBDDPGAgentConfig):
    # @package agent
    _target_: str = "url_benchmark.agent.discrete_fb.DiscreteFBAgent"
    name: str = "discrete_fb"
    preprocess: bool = False
    expl_eps: float = 0.2
    boltzmann = True
    temp = 100



cs = ConfigStore.instance()
cs.store(group="agent", name="discrete_fb", node=DiscreteFBAgentConfig)



class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim * self.action_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs = self.obs_action_net(obs)
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1.reshape(-1, self.z_dim, self.action_dim), F2.reshape(-1, self.z_dim, self.action_dim)

class DiscreteFBAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = DiscreteFBAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        # models
        if cfg.obs_type == 'pixels':
            self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
            self.encoder: nn.Module = Encoder(cfg.obs_shape).to(cfg.device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        goal_dim = self.obs_dim
        if cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(cfg.goal_space)
        if cfg.z_dim < goal_dim:
            logger.warning(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")

        self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                      cfg.feature_dim, cfg.hidden_dim,
                                      preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
            self.backward_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(
                cfg.device)
            self.backward_target_net = BackwardMap(goal_dim,
                                                   cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        self.forward_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                             cfg.feature_dim, cfg.hidden_dim,
                                             preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)

        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)

        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        # filter out small reward
        # pdb.set_trace()
        # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
        # obs = obs[idx]
        # reward = reward[idx]
        with torch.no_grad():
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        # self.solved_meta = meta
        return meta

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
            self,
            meta: MetaDict,
            global_step: int,
            time_step: TimeStep,
            finetune: bool = False,
            replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0 and np.random.rand() < self.cfg.update_z_proba:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        F1, F2 = self.forward_net(h, z)
        Q1, Q2 = [torch.einsum('sda, sd -> sa', Fi, z) for Fi in [F1, F2]]
        Q = torch.min(Q1, Q2)
        action = Q.max(1)[1]

        if not eval_mode:
            if step < self.cfg.num_expl_steps:
                action = torch.randint_like(action, self.action_dim)
            else:
                action = torch.randint_like(action, self.action_dim) \
                    if np.random.rand() < self.cfg.expl_eps else action
        return action.item()


    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            # compute greedy action
            target_F1, target_F2 = self.forward_target_net(next_obs, z)
            next_Q1, next_Q2 = [torch.einsum('sda, sd -> sa', Fi, z) for Fi in [target_F1, target_F2]]
            next_Q = torch.min(next_Q1, next_Q2)

            if self.cfg.boltzmann:
                pi = F.softmax(next_Q / self.cfg.temp, dim=-1)
                target_F1, target_F2 = [torch.einsum("sa, sda -> sd", pi, Fi) for Fi in [target_F1, target_F2]] # batch x z_dim
                next_Q = torch.einsum("sa, sa -> s", pi, next_Q)
            else:
                next_action = next_Q.max(1)[1]
                next_idx = next_action[:, None].repeat(1, self.cfg.z_dim)[:, :, None]
                target_F1, target_F2 = [Fi.gather(-1, next_idx).squeeze() for Fi in [target_F1, target_F2]]  # batch x z_dim
                next_Q = next_Q.max(1)[0]
            target_B = self.backward_target_net(next_goal)  # batch x z_dim
            target_M1, target_M2 = [torch.einsum('sd, td -> st', Fi, target_B) \
                                    for Fi in [target_F1, target_F2]] # batch x batch
            target_M = torch.min(target_M1, target_M2)

        # compute FB loss
        idxs = action.repeat(1, self.cfg.z_dim)[:, :, None]
        F1, F2 = [Fi.gather(-1, idxs).squeeze() for Fi in self.forward_net(obs, z)]
        B = self.backward_net(next_goal)
        M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
        M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
        I = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~I.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # Q LOSS

        if self.cfg.q_loss:
            with torch.no_grad():
                # next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
                # next_Q = torch.min(next_Q1, nextQ2)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.linalg.pinv(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size

            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            fb_loss += self.cfg.q_loss_coef * q_loss

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        # Cov = torch.cov(B.T)  # Vicreg loss
        # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
        # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
        # orth_loss =  var_loss + cov_loss
        # fb_loss += self.cfg.ortho_coef * orth_loss

        if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics


    def update_target_net(self):
        # update critic target for both forward and backward net
        utils.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.cfg.fb_target_tau)


    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        # pdb.set_trace()
        obs = batch.obs
        action = batch.action.type(torch.int64)
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        # if len(batch.meta) == 1 and batch.meta[0].shape[-1] == self.cfg.z_dim:
        #     z = batch.meta[0]
        #     invalid = torch.linalg.norm(z, dim=1) < 1e-15
        #     if sum(invalid):
        #         z[invalid, :] = self.sample_z(sum(invalid)).to(self.cfg.device)
        # else:
        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")
        # obs = self.aug_and_encode(batch.obs)
        # next_obs = self.aug_and_encode(batch.next_obs)
        # if not self.cfg.update_encoder:
        #     obs = obs.detach()
        #     next_obs = next_obs.detach()

        backward_input = batch.obs
        future_goal = batch.future_obs
        if self.cfg.goal_space is not None:
            assert batch.goal is not None
            backward_input = batch.goal
            future_goal = batch.future_goal

        perm = torch.randperm(self.cfg.batch_size)
        backward_input = backward_input[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            if not self.cfg.rand_weight:
                with torch.no_grad():
                    mix_z = self.backward_net(backward_input[mix_idxs]).detach()
            else:
                # generate random weight
                weight = torch.rand(size=(mix_idxs.shape[0], self.cfg.batch_size)).to(self.cfg.device)
                weight = F.normalize(weight, dim=1)
                uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.cfg.device)
                weight = uniform_rdv * weight
                with torch.no_grad():
                    mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
            if self.cfg.norm_z:
                mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
            z[mix_idxs] = mix_z

        # hindsight replay
        if self.cfg.future_ratio > 0:
            assert future_goal is not None
            future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)
            z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

        metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs, next_goal=next_goal, z=z, step=step))

        # update critic target
        # utils.soft_update_params(self.forward_net, self.forward_target_net,
        #                          self.cfg.fb_target_tau)
        # utils.soft_update_params(self.backward_net, self.backward_target_net,
        #                          self.cfg.fb_target_tau)

        return metrics
