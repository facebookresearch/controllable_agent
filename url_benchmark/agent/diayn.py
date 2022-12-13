# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp
from typing import Any, Dict, Tuple
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
from url_benchmark.dmc import TimeStep
from .ddpg import DDPGAgent, MetaDict, DDPGAgentConfig
from url_benchmark.in_memory_replay_buffer import ReplayBuffer


@dataclasses.dataclass
class DIAYNAgentConfig(DDPGAgentConfig):
    _target_: str = "url_benchmark.agent.diayn.DIAYNAgent"
    name: str = "diayn"
    update_encoder: bool = omegaconf.II("update_encoder")
    skill_dim: int = 16
    diayn_scale: float = 1.0
    update_skill_every_step: int = 50


cs = ConfigStore.instance()
cs.store(group="agent", name="diayn", node=DIAYNAgentConfig)


class DIAYN(nn.Module):
    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        skill_pred = self.skill_pred_net(obs)
        return skill_pred


class DIAYNAgent(DDPGAgent):
    def __init__(self, **kwargs) -> None:
        cfg = DIAYNAgentConfig(**kwargs)

        # create actor and critic
        # increase obs shape to include skill dim (through meta_dim)
        super().__init__(**kwargs, meta_dim=cfg.skill_dim)
        self.cfg = cfg  # override base ddpg cfg type

        # create diayn
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

        self.diayn.train()

    def init_meta(self) -> tp.Dict[str, np.ndarray]:
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
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
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_diayn(self, skill, next_obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        loss, df_accuracy = self.compute_diayn_loss(next_obs, skill)

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step) -> Any:
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        # TODO pred_z unused, is that normal?
        # _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)

        return reward * self.diayn_scale

    def compute_diayn_loss(self, next_state, skill) -> Tuple[Any, Any]:
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(next_state)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size).to(self.device)
        obs, action, extr_reward, discount, next_obs = batch.unpack()
        skill = batch.meta["skill"]

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_diayn(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

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
