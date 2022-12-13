# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import pdb
import typing as tp
import torch

from url_benchmark import utils
from .ddpg import DDPGAgent
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
import dataclasses
from url_benchmark.agent.ddpg import DDPGAgentConfig
from hydra.core.config_store import ConfigStore
import omegaconf


@dataclasses.dataclass
class MaxEntAgentConfig(DDPGAgentConfig):
    _target_: str = "url_benchmark.agent.max_ent.MaxEntAgent"
    name: str = "max_ent"
    knn_rms: bool = True
    knn_k: int = 12
    knn_avg: bool = True
    knn_clip: float = 0.0001
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")

cs = ConfigStore.instance()
cs.store(group="agent", name="max_ent", node=MaxEntAgentConfig)

class MaxEntAgent(DDPGAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = MaxEntAgentConfig(**kwargs)
        self.cfg = cfg
        # particle-based entropy
        rms = utils.RMS(self.cfg.device)
        self.pbe = utils.PBE(rms, self.cfg.knn_clip, self.cfg.knn_k, self.cfg.knn_avg, cfg.knn_rms,
                             self.cfg.device)

    def compute_intr_reward(self, goal: torch.Tensor, step: int) -> torch.Tensor:
        reward = self.pbe(goal)
        intr_ent_reward = reward.reshape(-1, 1)
        return intr_ent_reward

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        obs = batch.obs
        action = batch.action
        discount = batch.discount

        next_goal = next_obs = batch.next_obs
        if self.cfg.goal_space is not None: # type: ignore
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        with torch.no_grad():
            reward = self.compute_intr_reward(goal=next_goal, step=step)

        if self.use_tb or self.use_wandb:
            metrics['intr_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount,
                               next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

