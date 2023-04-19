# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import dataclasses
from typing import Any, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark import utils
from .fb_modules import mlp

# MetaDict = tp.Mapping[str, tp.Union[np.ndarray, torch.Tensor]]
MetaDict = tp.Mapping[str, np.ndarray]


@dataclasses.dataclass
class DDPGAgentConfig:
    _target_: str = "url_benchmark.agent.ddpg.DDPGAgent"
    name: str = "ddpg"
    reward_free: bool = omegaconf.II("reward_free")
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")
    use_wandb: bool = omegaconf.II("use_wandb")
    num_expl_steps: int = omegaconf.MISSING  # to be specified later
    hidden_dim: int = 1024
    feature_dim: int = 50
    stddev_schedule: float = 0.2
    stddev_clip: float = 0.3
    nstep: int = 3
    batch_size: int = 1024  # 256 for pixels
    init_critic: bool = True
    # update_encoder: ${update_encoder}  # not in the config


cs = ConfigStore.instance()
cs.store(group="agent", name="ddpg", node=DDPGAgentConfig)


class Encoder(nn.Module):
    def __init__(self, obs_shape) -> None:
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim) -> None:
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std) -> utils.TruncatedNormal:
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim) -> None:
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action) -> Tuple[Any, Any]:
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
    # pylint: disable=unused-argument
    def __init__(self, meta_dim: int = 0, **kwargs: tp.Any) -> None:
        if self.__class__.__name__.startswith(("DIAYN", "APS", "RND", "Proto", "ICMAPT", "MaxEnt", "Exploration")):  # HACK
            cfg_fields = {field.name for field in dataclasses.fields(DDPGAgentConfig)}
            # those have their own config, so lets curate the fields
            # others will need to be ported in time
            kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        cfg = DDPGAgentConfig(**kwargs)
        self.cfg = cfg
        self.action_dim = cfg.action_shape[0]
        self.solved_meta = None
        # self.update_encoder = update_encoder  # used in subclasses

        # models
        if cfg.obs_type == 'pixels':
            self.aug: tp.Union[utils.RandomShiftsAug, nn.Identity] = utils.RandomShiftsAug(pad=4)
            self.encoder: tp.Union[Encoder, nn.Identity] = Encoder(cfg.obs_shape).to(cfg.device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0] + meta_dim

        self.actor = Actor(cfg.obs_type, self.obs_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim).to(cfg.device)

        self.critic: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.reward_model: tp.Optional[torch.nn.Module] = None
        self.reward_opt: tp.Optional[torch.optim.Adam] = None
        if self.reward_free:
            self.reward_model = mlp(self.obs_dim, cfg.hidden_dim, "ntanh", cfg.hidden_dim,  # type: ignore
                                    "relu", cfg.hidden_dim, "relu", 1).to(cfg.device)  # type: ignore
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        self.train()
        self.critic_target.train()

    def __getattr__(self, name: str) -> tp.Any:
        # LEGACY: allow accessing the config directly as attribute
        # to avoid having to rewrite everything at once
        # cost: less type safety
        if "cfg" in self.__dict__:
            return getattr(self.cfg, name)
        raise AttributeError

    def train(self, training: bool = True) -> None:
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def init_meta(self) -> MetaDict:
        return OrderedDict()

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        return meta

    def act(self, obs, meta, step, eval_mode) -> np.ndarray:
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def train_reward(self, replay_loader: ReplayBuffer) -> None:
        obs_list, reward_list = [], []
        batch_size = 0
        num_inference_steps = 10000
        while batch_size < num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            obs, action, reward, discount, next_obs = batch.to(self.device).unpack()
            del obs, action, discount
            obs_list.append(next_obs)
            reward_list.append(reward)
            batch_size += next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[: num_inference_steps], reward[: num_inference_steps]
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])
        assert self.reward_model is not None
        for i in range(2000):
            reward_loss = (self.reward_model(obs) - reward).pow(2).mean()
            assert self.reward_opt is not None
            self.reward_opt.zero_grad(set_to_none=True)
            reward_loss.backward()
            self.reward_opt.step()
            print(f"iteration: {i}, reward_loss: {reward_loss.item()}")

        # compute test loss:
        while batch_size < num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            obs, action, reward, discount, next_obs = batch.to(self.device).unpack()
            del obs, action, discount
            obs_list.append(next_obs)
            reward_list.append(reward)
            batch_size += next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[: num_inference_steps], reward[: num_inference_steps]
        test_loss = (self.reward_model(obs) - reward).pow(2).mean()
        print(f"Test Loss: {test_loss.item()}")

    @tp.no_type_check  # TODO remove
    def update_critic(self, obs, action, reward, discount, next_obs, step) -> tp.Dict[str, float]:
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    @tp.no_type_check  # TODO remove
    def update_actor(self, obs, step) -> tp.Dict[str, float]:
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs) -> Any:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        obs, action, reward, discount, next_obs, *_ = batch.to(self.device).unpack()
        if self.reward_free:
            del reward
            assert self.reward_model is not None
            reward = self.reward_model(next_obs)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
