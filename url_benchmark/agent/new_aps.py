# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import typing as tp
import dataclasses
from collections import OrderedDict
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from url_benchmark import utils
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from .ddpg import MetaDict
from .ddpg import Encoder
from .fb_modules import Actor, ForwardMap, mlp
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark import goals as _goals

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class APSAgentConfig:
    _target_: str = "url_benchmark.agent.new_aps.NEWAPSAgent"
    name: str = "new_aps"
    reward_free: bool = omegaconf.II("reward_free")
    custom_reward: tp.Optional[str] = omegaconf.II("custom_reward")
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    sf_target_tau: float = 0.01
    update_every_steps: float = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING
    hidden_dim: int = 1024
    feature_dim: int = 512
    backward_hidden_dim: int = 512
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # "0.2"
    stddev_clip: str = "0.3"  # 1
    nstep: int = 1
    batch_size: int = 512  # 256 for pixels
    init_critic: bool = True
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    preprocess: bool = False
    update_encoder: bool = omegaconf.II("update_encoder")
    z_dim: int = 10
    update_z_every_step: int = 100
    knn_rms: bool = True
    knn_k: int = 12
    knn_avg: bool = True
    knn_clip: float = 0.0001
    num_init_steps: int = 4096  # set to ${num_train_frames} to disable finetune policy parameters
    num_inference_steps: int = 5120
    add_trunk: bool = False
    lr_coef: float = 1
    future_ratio: float = 0


cs = ConfigStore.instance()
cs.store(group="agent", name="new_aps", node=APSAgentConfig)


class FeatureNet(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, norm=True):
        phi = self.net(obs)
        return F.normalize(phi, dim=1) if norm else phi


class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net = FeatureNet(obs_dim, z_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        """MLE loss"""
        phi = self.feature_net(obs)
        loss = -torch.einsum("bd,bd->b", phi, z).mean()
        return loss


class NEWAPSAgent:
    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ) -> None:
        cfg = APSAgentConfig(**kwargs)
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

        self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim,
                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.successor_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # build up the target network
        self.successor_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)

        self.feature_learner = FeatureLearner(goal_dim, cfg.z_dim, cfg.backward_hidden_dim).to(cfg.device)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr)
        self.train()
        self.successor_target_net.train()

        # particle-based entropy
        rms = utils.RMS(self.cfg.device)
        self.pbe = utils.PBE(rms, cfg.knn_clip, cfg.knn_k, cfg.knn_avg, cfg.knn_rms,
                             cfg.device)

        self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.successor_net, self.feature_learner]:
            net.train(training)

    def sample_z(self, size):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = F.normalize(gaussian_rdv, dim=1)
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
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def precompute_cov(self, replay_loader: ReplayBuffer) -> None:
        print("computing Cov of phi to be used at inference")
        obs_list = list()
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs = batch.next_goal if self.cfg.goal_space is not None else batch.next_obs
            if obs is None:
                raise ValueError("Obs should never be None")
            obs_list.append(obs)
            batch_size += batch.next_obs.size(0)
        obs = torch.cat(obs_list, 0)
        self.inv_cov = self._compute_cov(obs)
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

        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.feature_learner.feature_net(desired_goal)
        z = torch.matmul(z, self.inv_cov)  # 1 x z_dim
        z = F.normalize(z, dim=1)
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

        with torch.no_grad():
            phi = self.feature_learner.feature_net(obs)
        z = torch.linalg.lstsq(phi, reward).solution  # z_dim x 1
        z = F.normalize(z, dim=0)  # be careful to the dimension
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        # self.solved_meta = meta
        return meta

    def update_phi(self, obs, z, step) -> tp.Dict[str, tp.Any]:
        metrics: tp.Dict[str, float] = {}
        loss = self.feature_learner(obs, z)
        self.phi_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.phi_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['phi_loss'] = loss.item()

        return metrics

    def compute_intrinsic_reward(self, next_obs, z, step) -> tp.Tuple[tp.Any, tp.Any]:
        # maxent reward
        with torch.no_grad():
            phi = self.feature_learner.feature_net(next_obs, norm=False)
        reward = self.pbe(phi)
        entropy_reward = reward.reshape(-1, 1)

        # successor feature reward
        phi = F.normalize(phi, dim=1)
        diayn_reward = torch.einsum("bi,bi->b", phi, z).reshape(-1, 1)

        return entropy_reward, diayn_reward

    def update_critic(self,
                      obs: torch.Tensor,
                      action: torch.Tensor,
                      reward: torch.Tensor,
                      discount: torch.Tensor,
                      next_obs: torch.Tensor,
                      z: torch.Tensor,
                      step: int) -> tp.Dict[str, tp.Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}

        # compute target critic
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
            next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
            next_Q = torch.min(next_Q1, next_Q2)
            target_Q = reward + discount * next_Q.reshape(-1, 1)
            target_Q = target_Q.squeeze(1)

        F1, F2 = self.successor_net(obs, z, action)
        Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
        sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        # sf_loss /= self.cfg.z_dim

        if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
            metrics['target_Q'] = target_Q.mean().item()
            metrics['Q1'] = Q1.mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['sf_loss'] = sf_loss.item()

        # optimize SF
        self.sf_opt.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_opt.step()

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        obs = batch.obs
        action = batch.action
        discount = batch.discount
        reward = batch.reward
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        z = batch.meta["z"]
        assert z.shape[-1] == self.cfg.z_dim

        if self.cfg.reward_free:
            # freeze successor features at finetuning phase
            metrics.update(self.update_phi(next_goal, z, step))

            with torch.no_grad():
                entropy_reward, diayn_reward = self.compute_intrinsic_reward(next_goal, z, step)
                intrinsic_reward = entropy_reward + diayn_reward

            if self.cfg.use_tb or self.cfg.use_wandb:
                metrics['intrinsic_reward'] = intrinsic_reward.mean().item()
                metrics['entropy_reward'] = entropy_reward.mean().item()
                metrics['diayn_reward'] = diayn_reward.mean().item()

            reward = intrinsic_reward

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['extrinsic_reward'] = batch.reward.mean().item()

        # hindsight replay
        if self.cfg.future_ratio > 0:
            future_goal = batch.future_goal if self.cfg.goal_space else batch.future_obs
            assert future_goal is not None
            future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)
            with torch.no_grad():
                phi = self.feature_learner.feature_net(future_goal)
            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)
            new_z = phi[future_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = F.normalize(new_z, dim=1)
            z[future_idxs] = new_z

        # update critic
        metrics.update(
            self.update_critic(obs=obs, action=action, reward=reward, discount=discount,
                               next_obs=next_obs, z=z, step=step))

        # update actor
        metrics.update(self.update_actor(obs=obs, z=z, step=step))

        # update critic target
        utils.soft_update_params(self.successor_net, self.successor_target_net,
                                 self.cfg.sf_target_tau)

        return metrics
