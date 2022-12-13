# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb # pylint: disable=unused-import
import typing as tp
from typing import Any, Dict
import copy
import dataclasses
import torch
from torch import nn
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
from .ddpg import DDPGAgent, DDPGAgentConfig
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
import url_benchmark.goals as _goals

@dataclasses.dataclass
class RNDAgentConfig(DDPGAgentConfig):
    _target_: str = "url_benchmark.agent.rnd.RNDAgent"
    name: str = "rnd"
    rnd_rep_dim: int = 512
    rnd_scale: float = 1.0
    update_encoder: bool = omegaconf.II("update_encoder")
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")


cs = ConfigStore.instance()
cs.store(group="agent", name="rnd", node=RNDAgentConfig)


class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.) -> None:
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs: nn.Module = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error


class RNDAgent(DDPGAgent):
    def __init__(self, **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        cfg = RNDAgentConfig(**kwargs)
        self.cfg = cfg

        goal_dim = self.obs_dim
        if self.cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(self.cfg.goal_space)

        self.rnd = RND(goal_dim, cfg.hidden_dim, cfg.rnd_rep_dim,
                       self.encoder, self.aug, (goal_dim, ),
                       cfg.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.lr)

        self.rnd.train()

    # pylint: disable=unused-argument
    def update_rnd(self, obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}
        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['rnd_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, step) -> Any:
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        goal = obs = batch.obs
        if self.cfg.goal_space is not None: # type: ignore
            assert batch.goal is not None
            goal = batch.goal

        action = batch.action
        extr_reward = batch.reward
        discount = batch.discount
        next_obs = batch.next_obs

        # update RND first
        if self.reward_free:
            # note: one difference is that the RND module is updated off policy
            metrics.update(self.update_rnd(goal, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(goal, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

            metrics['pred_error_mean'] = self.intrinsic_reward_rms.M.item()
            metrics['pred_error_std'] = torch.sqrt(self.intrinsic_reward_rms.S).item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
