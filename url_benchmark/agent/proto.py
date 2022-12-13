# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import dataclasses
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
from .ddpg import DDPGAgent
from .ddpg import DDPGAgentConfig as _BaseConfig
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from typing import Any, Dict, TypeVar

_T = TypeVar('_T')


@jit.script
def sinkhorn_knopp(Q):
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
    for _ in range(3):
        u = Q.sum(dim=1)
        u = r / u
        Q *= u.unsqueeze(dim=1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
    Q = Q / Q.sum(dim=0, keepdim=True)
    return Q.T


class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim) -> None:
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(pred_dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, pred_dim))

        self.apply(utils.weight_init)

    def forward(self, x) -> Any:
        return self.trunk(x)


@dataclasses.dataclass
class ProtoAgentConfig(_BaseConfig):
    _target_: str = "url_benchmark.agent.proto.ProtoAgent"
    name: str = "proto"
    update_encoder: bool = omegaconf.II("update_encoder")
    pred_dim: int = 128
    proj_dim: int = 512
    num_protos: int = 512
    tau: float = 0.1
    topk: int = 3
    queue_size: int = 2048
    encoder_target_tau: float = 0.05


cs = ConfigStore.instance()
cs.store(group="agent", name="proto", node=ProtoAgentConfig)


class ProtoAgent(DDPGAgent):
    def __init__(self, **kwargs) -> None:
        cfg = ProtoAgentConfig(**kwargs)
        super().__init__(**kwargs)
        self.cfg = cfg  # override base ddpg cfg type

        # models
        self.encoder_target = deepcopy(self.encoder)

        self.predictor = nn.Linear(self.obs_dim, cfg.pred_dim).to(self.device)
        self.predictor.apply(utils.weight_init)
        self.predictor_target = deepcopy(self.predictor)

        self.projector = Projector(cfg.pred_dim, cfg.proj_dim).to(self.device)
        self.projector.apply(utils.weight_init)

        # prototypes
        self.protos = nn.Linear(cfg.pred_dim, cfg.num_protos,
                                bias=False).to(self.device)
        self.protos.apply(utils.weight_init)

        # candidate queue
        self.queue = torch.zeros(cfg.queue_size, cfg.pred_dim, device=self.device)
        self.queue_ptr = 0

        # optimizers
        self.proto_opt = torch.optim.Adam(utils.chain(
            self.encoder.parameters(), self.predictor.parameters(),
            self.projector.parameters(), self.protos.parameters()),
            lr=self.lr)

        self.predictor.train()
        self.projector.train()
        self.protos.train()

    def init_from(self, other) -> None:
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        utils.hard_update_params(other.protos, self.protos)
        if self.init_critic:
            utils.hard_update_params(other.critic, self.critic)

    def normalize_protos(self) -> None:
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

    # pylint: disable=unused-argument
    def compute_intr_reward(self, obs, step) -> Any:
        self.normalize_protos()
        # find a candidate for each prototype
        with torch.no_grad():
            z = self.encoder(obs)
            z = self.predictor(z)
            z = F.normalize(z, dim=1, p=2)
            scores = self.protos(z).T
            prob = F.softmax(scores, dim=1)
            candidates = pyd.Categorical(prob).sample()

        # enqueue candidates
        ptr = self.queue_ptr
        self.queue[ptr:ptr + self.num_protos] = z[candidates]
        self.queue_ptr = (ptr + self.num_protos) % self.queue.shape[0]

        # compute distances between the batch and the queue of candidates
        z_to_q = torch.norm(z[:, None, :] - self.queue[None, :, :], dim=2, p=2)
        all_dists, _ = torch.topk(z_to_q, self.topk, dim=1, largest=False)
        dist = all_dists[:, -1:]
        reward = dist
        return reward

    # pylint: disable=unused-argument
    def update_proto(self, obs, next_obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        # normalize prototypes
        self.normalize_protos()

        # online network
        s = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.tau, dim=1)

        # target network
        with torch.no_grad():
            t = self.encoder_target(next_obs)
            t = self.predictor_target(t)
            t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)
            q_t = sinkhorn_knopp(scores_t / self.tau)

        # loss
        loss = -(q_t * log_p_s).sum(dim=1).mean()
        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        self.proto_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.proto_opt.step()

        return metrics

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        obs, action, extr_reward, discount, next_obs = batch.to(self.device).unpack()

        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)

        if self.reward_free:
            metrics.update(self.update_proto(obs, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        obs = self.encoder(obs)
        next_obs = self.encoder(next_obs)

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
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
