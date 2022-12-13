# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import typing as tp
import dataclasses
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from url_benchmark import utils
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from .ddpg import DDPGAgent, MetaDict, DDPGAgentConfig
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from typing import Any, Dict, Tuple

# TODO(HL): how to include GPI for continuous domain?


@dataclasses.dataclass
class APSAgentConfig(DDPGAgentConfig):
    _target_: str = "url_benchmark.agent.aps.APSAgent"
    name: str = "aps"
    update_encoder: bool = omegaconf.II("update_encoder")
    sf_dim: int = 10
    update_task_every_step: int = 5
    knn_rms: bool = True
    knn_k: int = 12
    knn_avg: bool = True
    knn_clip: float = 0.0001
    num_init_steps: int = 4096  # set to ${num_train_frames} to disable finetune policy parameters
    lstsq_batch_size: int = 4096
    num_inference_steps: int = 10000


cs = ConfigStore.instance()
cs.store(group="agent", name="aps", node=APSAgentConfig)


class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 sf_dim) -> None:
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
            q_layers += [nn.Linear(hidden_dim, sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action, task) -> Tuple[Any, Any]:
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        q1 = torch.einsum("bi,bi->b", task, q1).reshape(-1, 1)
        q2 = torch.einsum("bi,bi->b", task, q2).reshape(-1, 1)

        return q1, q2


class APS(nn.Module):
    def __init__(self, obs_dim, sf_dim, hidden_dim) -> None:
        super().__init__()
        self.state_feat_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, sf_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, norm=True) -> Any:
        state_feat = self.state_feat_net(obs)
        state_feat = F.normalize(state_feat, dim=-1) if norm else state_feat
        return state_feat


class APSAgent(DDPGAgent):
    def __init__(self, **kwargs: tp.Any) -> None:
        cfg = APSAgentConfig(**kwargs)

        # create actor and critic
        # increase obs shape to include task dim (through meta_dim)
        super().__init__(**kwargs, meta_dim=cfg.sf_dim)
        self.cfg: APSAgentConfig = cfg  # override base ddpg cfg type

        # overwrite critic with critic sf
        self.critic = CriticSF(cfg.obs_type, self.obs_dim, self.action_dim,
                               self.feature_dim, self.hidden_dim,
                               self.sf_dim).to(self.device)
        self.critic_target = CriticSF(self.obs_type, self.obs_dim,
                                      self.action_dim, self.feature_dim,
                                      self.hidden_dim,
                                      self.sf_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.lr)

        self.aps = APS(self.obs_dim - self.sf_dim, self.sf_dim,
                       kwargs['hidden_dim']).to(kwargs['device'])

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, cfg.knn_clip, cfg.knn_k, cfg.knn_avg, cfg.knn_rms,
                             cfg.device)

        # optimizers
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

        self.aps.train()

    def init_meta(self) -> tp.Dict[str, np.ndarray]:
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task_array = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task_array
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
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def update_aps(self, task, next_obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        loss = self.compute_aps_loss(next_obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.aps_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['aps_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, task, next_obs, step) -> Tuple[Any, Any]:
        # maxent reward
        with torch.no_grad():
            rep = self.aps(next_obs, norm=False)
        reward = self.pbe(rep)
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward, intr_sf_reward

    def compute_aps_loss(self, next_obs, task) -> Any:
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", task, self.aps(next_obs)).mean()
        return loss

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size).to(self.device)

        obs, action, extr_reward, discount, next_obs = batch.unpack()
        task = batch.meta["task"]

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            # freeze successor features at finetuning phase
            metrics.update(self.update_aps(task, next_obs, step))

            with torch.no_grad():
                intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                    task, next_obs, step)
                intr_reward = intr_ent_reward + intr_sf_reward

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                metrics['intr_sf_reward'] = intr_sf_reward.mean().item()

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), task, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), task, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    @torch.no_grad()
    def regress_meta(self, replay_loader, step):
        obs, reward = [], []
        batch_size = 0
        while batch_size < self.lstsq_batch_size:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch_obs, _, batch_reward, *_ = utils.to_torch(batch, self.device)
            obs.append(batch_obs)
            reward.append(batch_reward)
            batch_size += batch_obs.size(0)
        obs, reward = torch.cat(obs, 0), torch.cat(reward, 0)

        obs = self.aug_and_encode(obs)
        rep = self.aps(obs)
        task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # save for evaluation
        self.solved_meta = meta
        return meta

    @torch.no_grad()
    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    @torch.no_grad()
    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        rep = self.aps(obs)
        # task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        task = torch.linalg.lstsq(rep, reward)[0].squeeze()
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # self.solved_meta = meta
        return meta

    def update_critic(self, obs, action, reward, discount, next_obs, task,
                      step) -> Dict[str, Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,
                                                      task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, task, step) -> Dict[str, Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, task)
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
