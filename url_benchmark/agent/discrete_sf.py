# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import dataclasses
import typing as tp
import torch
from hydra.core.config_store import ConfigStore


from .sf import SFAgent, SFAgentConfig


@dataclasses.dataclass
class DiscreteSFAgentConfig(SFAgentConfig):
    # @package agent
    _target_: str = "url_benchmark.agent.discrete_sf.DiscreteSFAgent"
    name: str = "discrete_sf"


cs = ConfigStore.instance()
cs.store(group="agent", name="discrete_sf", node=DiscreteSFAgentConfig)


class DiscreteSFAgent(SFAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        num_xy = 5
        x = y = torch.linspace(-1, 1, num_xy, dtype=torch.float32, device=self.cfg.device)
        XX, YY = torch.meshgrid(x, y)
        X = XX.reshape(-1, 1)
        Y = YY.reshape(-1, 1)
        self.ACTION_GRID = torch.cat([X, Y], dim=1)

    def greedy_action(self, obs, z):
        OBS = obs.repeat(1, self.ACTION_GRID.shape[0]).reshape(self.ACTION_GRID.shape[0] * obs.shape[0], obs.shape[1])
        Z = z.repeat(1, self.ACTION_GRID.shape[0]).reshape(self.ACTION_GRID.shape[0] * z.shape[0], z.shape[1])
        ACTION = self.ACTION_GRID.repeat(obs.shape[0], 1)

        F1, F2 = self.successor_net(OBS, Z, ACTION)
        Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, Z) for Fi in [F1, F2]]
        Q = torch.min(Q1, Q2)
        max_idx = Q.reshape(obs.shape[0], self.ACTION_GRID.shape[0]).max(dim=1)[1]
        return self.ACTION_GRID[max_idx]

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0)  # type: ignore
        obs = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        action = self.greedy_action(obs, z)
        if not eval_mode:
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        return metrics

    def update_sf(  # type: ignore
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            next_action = self.greedy_action(next_obs, z)
            target_F1, target_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
            target_phi = self.feature_learner.feature_net(next_goal).detach()  # batch x z_dim
            target_F = torch.min(target_F1, target_F2)
            target_F = target_phi + discount * target_F

        # compute SF loss
        F1, F2 = self.successor_net(obs, z, action)
        # sf_loss = torch.norm(F1 - target_F, dim=-1, p='fro').mean()
        # sf_loss += torch.norm(F2 - target_F, dim=-1, p='fro').mean()
        sf_loss = (F1 - target_F).pow(2).mean()
        sf_loss += (F2 - target_F).pow(2).mean()

        # compute feature loss
        phi_loss = self.feature_learner(obs=obs, action=action, next_obs=next_obs, future_obs=future_obs)

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
