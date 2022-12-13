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
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, mlp, OnlineCov
from dm_env import specs
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SFSVDAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.sf_svd.SFSVDAgent"
    name: str = "sf_svd"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 5
    sf_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 5120
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 512   # 128, 2048
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 100  # 30-200
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # 0,  0.1, 0.2
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 100
    nstep: int = 1
    batch_size: int = 1024
    init_sf: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    # ortho_coef: float = 0.1  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    preprocess: bool = True
    num_sf_updates: int = 1
    feature_learner: str = "p"
    mix_ratio: float = 0.0
    q_loss: bool = True
    update_cov_every_step: int = 1000
    add_trunk: bool = False


cs = ConfigStore.instance()
cs.store(group="agent", name="sf_svd", node=SFSVDAgentConfig)


class SVDLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net = mlp(obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(torch.cat([obs, action], dim=1))
        mu = self.mu_net(next_obs)
        P = torch.einsum("sd, td -> st", phi, mu)
        I = torch.eye(*P.size(), device=P.device)
        off_diag = ~I.bool()
        loss = - 2 * P.diag().mean() + P[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class SFSVDAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = SFSVDAgentConfig(**kwargs)
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
            g = next(iter(_goals.goals.funcs[cfg.goal_space].values()))()
            assert len(g.shape) == 1
            goal_dim = len(g)
        if cfg.z_dim < goal_dim:
            logger.warning(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(cfg.obs_type, self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        else:
            self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                               cfg.feature_dim, cfg.hidden_dim,
                               preprocess=cfg.preprocess, add_trunk=cfg.add_trunk).to(cfg.device)
        self.successor_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=cfg.add_trunk).to(cfg.device)
        # build up the target network
        self.successor_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=cfg.add_trunk).to(cfg.device)

        self.feature_learner = SVDLearner(goal_dim, self.action_dim, cfg.z_dim, cfg.backward_hidden_dim).to(cfg.device)
        # if cfg.debug:
        #     self.feature_learner: nn.Module = IdentityMap().to(cfg.device)

        # self.feature_net = BackwardMap(cfg.obs_type, goal_dim, cfg.z_dim, cfg.backward_hidden_dim).to(cfg.device)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt: tp.Optional[torch.optim.Adam] = None
        self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.train()
        self.successor_target_net.train()

        self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.successor_net]:
            net.train(training)
        if self.phi_opt is not None:
            self.feature_learner.train()

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_sf:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def precompute_cov(self, replay_loader: ReplayBuffer) -> None:
        print("computing Cov of phi to be used at inference")
        obs_list = []
        action_list = []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs = batch.goal if self.cfg.goal_space is not None else batch.obs
            if obs is None:
                raise ValueError("Obs should never be None")
            obs_list.append(obs)
            action_list.append(batch.action)
            batch_size += batch.next_obs.size(0)
        obs = torch.cat(obs_list, 0)
        action = torch.cat(action_list, 0)
        self.inv_cov = self._compute_cov(torch.cat([obs, action], dim=1))
        # with torch.no_grad():
        #     phi = self.feature_learner.feature_net(obs)
        # cov = torch.matmul(phi.T, phi) / phi.shape[0]
        # self.inv_cov = torch.linalg.pinv(cov)

    def _compute_cov(self, goal: torch.Tensor) -> torch.Tensor:
        # compute inverse of cov of phi
        with torch.no_grad():
            phi = self.feature_learner.feature_net(goal)
        cov = torch.matmul(phi.T, phi) / phi.shape[0]
        inv_cov = torch.inverse(cov)
        return inv_cov

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        # assert self.cfg.feature_learner in ["FB"]

        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        dummy_action = torch.zeros((1, self.action_dim), dtype=torch.float32).to(self.cfg.device)
        with torch.no_grad():
            z = self.feature_learner.feature_net(torch.cat([desired_goal, dummy_action], dim=1))

        z = torch.matmul(z, self.inv_cov)  # 1 x z_dim
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list, action_list = [], [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.goal if self.cfg.goal_space is not None else batch.obs)
            action_list.append(batch.action)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward, action = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(action_list, 0)  # type: ignore
        obs, reward, action = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps], action[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_action_and_rewards(obs, action, reward)

    def infer_meta_from_obs_action_and_rewards(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        with torch.no_grad():
            phi = self.feature_learner.feature_net(torch.cat([obs, action], dim=1))
        z = torch.linalg.lstsq(phi, reward).solution  # z_dim x 1
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)  # be careful to the dimension
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        # self.solved_meta = meta
        return meta

    def sample_z(self, size):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(gaussian_rdv, dim=1)
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
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_sf(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        future_goal: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            if self.cfg.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.cfg.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
            target_phi = self.feature_learner.feature_net(torch.cat([goal, action], dim=1)).detach()  # batch x z_dim
            next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F

        F1, F2 = self.successor_net(obs, z, action)
        if not self.cfg.q_loss:
            # compute SF loss
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)
        else:
            # alternative loss
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            target_Q = torch.einsum('sd, sd -> s', target_F, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            sf_loss /= self.cfg.z_dim

        # compute feature loss
        phi_loss = self.feature_learner(obs=goal, action=action, next_obs=next_goal, future_obs=future_goal)

        if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
            metrics['target_F'] = target_F.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['phi'] = target_phi.mean().item()
            metrics['phi_norm'] = torch.norm(target_phi, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['sf_loss'] = sf_loss.item()
            if phi_loss is not None:
                metrics['phi_loss'] = phi_loss.item()

            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]
            # if self.cfg.goal_space in ["simplified_walker", "simplified_quadruped"]:
            #     metrics['max_velocity'] = goal_prime[:, -1].max().item()

        # optimize SF
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.sf_opt.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        # optimise phi
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward()
            self.phi_opt.step()

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics
        goal_list = []
        for _ in range(self.cfg.num_sf_updates):
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs = goal = batch.obs
            action = batch.action
            discount = batch.discount
            next_obs = next_goal = batch.next_obs
            future_goal = batch.future_obs
            if self.cfg.goal_space:
                assert batch.goal is not None
                assert batch.next_goal is not None
                goal = batch.goal
                next_goal = batch.next_goal
                future_goal = batch.future_goal

            goal_list.append(next_goal)

            z = self.sample_z(self.cfg.batch_size).to(self.cfg.device)
            if not z.shape[-1] == self.cfg.z_dim:
                raise RuntimeError("There's something wrong with the logic here")

            if self.cfg.mix_ratio > 0:
                perm = torch.randperm(self.cfg.batch_size)
                desired_goal = next_goal[perm]
                dummy_action = torch.zeros((desired_goal.shape[0], self.action_dim), dtype=torch.float32).to(self.cfg.device)
                with torch.no_grad():
                    phi = self.feature_learner.feature_net(torch.cat([desired_goal, dummy_action], dim=1))
                # compute inverse of cov of phi
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.inverse(cov)

                mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]

                new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
                new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
                z[mix_idxs] = new_z

            metrics.update(self.update_sf(obs=obs, goal=goal, action=action, discount=discount,
                                          next_obs=next_obs, next_goal=next_goal, future_goal=future_goal,
                                          z=z, step=step))

            # update actor
            metrics.update(self.update_actor(obs, z, step))

            # update critic target
            utils.soft_update_params(self.successor_net, self.successor_target_net,
                                     self.cfg.sf_target_tau)

        # update inv cov
        # if step % self.cfg.update_cov_every_step == 0:
        #     logger.info("update online cov")
        #     obs_list = list()
        #     batch_size = 0
        #     while batch_size < 10000:
        #         batch = next(replay_loader)
        #         batch = batch.to(self.cfg.device)
        #         obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
        #         batch_size += batch.next_obs.size(0)
        #     obs = torch.cat(obs_list, 0)
        #     with torch.no_grad():
        #         phi = self.feature_learner.feature_net(obs)
        #     self.inv_cov = torch.inverse(self.online_cov(phi))

        return metrics
