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
import omegaconf
from dm_env import specs

from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, OnlineCov


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UVFAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.uvf.UVFAgent"
    name: str = "uvf"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
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
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 512  # multiple de 3, 500-5000
    init_fb: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    additional_metric: bool = False
    add_trunk: bool = False


cs = ConfigStore.instance()
cs.store(group="agent", name="uvf", node=UVFAgentConfig)


class UVFAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = UVFAgentConfig(**kwargs)
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
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        else:
            self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                               cfg.feature_dim, cfg.hidden_dim,
                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                      cfg.feature_dim, cfg.hidden_dim,
                                      preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            # self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
            self.backward_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
            # self.backward_target_net = BackwardMap(goal_dim,
            #                                        cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        self.forward_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                             cfg.feature_dim, cfg.hidden_dim,
                                             preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        # self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
        # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)

        self.train()
        self.forward_target_net.train()
        # self.backward_target_net.train()
        self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
        # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "forward_target_net"] # + ["backward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        # if self.cfg.norm_z:
        #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
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


    def init_meta(self, replay_loader: tp.Optional[ReplayBuffer] = None) -> MetaDict:
        if replay_loader is not None:
            batch = replay_loader.sample(self.cfg.batch_size)
            assert batch.next_goal is not None
            g = batch.next_goal[0]
            meta = self.get_goal_meta(g)
        else:
            z = np.zeros((self.cfg.z_dim,), dtype=np.float32)
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


    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        desired_goal: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        # Q LOSS
        epsilon = 1e-6
        z = self.backward_net(desired_goal)
        reward =  (torch.norm(next_goal - desired_goal, dim=1, keepdim=False) < epsilon).float() # batch_size
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
            next_Q = torch.min(next_Q1, nextQ2)
            target_Q = reward + discount.squeeze(1) * next_Q  # batch_size
            target_Q = target_Q.detach()
        F1, F2 = self.forward_net(obs, z, action)
        Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
        fb_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
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

    def update_actor(self, obs: torch.Tensor, desired_goal: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        z = self.backward_net(desired_goal)
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net(obs, z, action)
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
            metrics['q'] = Q.mean().item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics


    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        # pdb.set_trace()
        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        # second_batch = replay_loader.sample(self.cfg.batch_size)
        # second_batch = second_batch.to(self.cfg.device)
        #
        # desired_goal = second_batch.next_obs
        # if self.cfg.goal_space is not None:
        #     assert second_batch.next_goal is not None
        #     desired_goal = second_batch.next_goal

        perm = torch.randperm(self.cfg.batch_size)
        desired_goal = next_goal[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            desired_goal[mix_idxs] = next_goal[mix_idxs]

        metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs, next_goal=next_goal, desired_goal=desired_goal, step=step))

        # update actor
        metrics.update(self.update_actor(obs, desired_goal, step))

        # update critic target
        utils.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.cfg.fb_target_tau)
        # utils.soft_update_params(self.backward_net, self.backward_target_net,
        #                          self.cfg.fb_target_tau)

        return metrics
