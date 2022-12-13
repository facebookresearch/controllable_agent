# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import typing as tp
from collections import OrderedDict
import dataclasses
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark import utils
from .fb_modules import mlp, Actor
from url_benchmark import goals as _goals
from pathlib import Path

logger = logging.getLogger(__name__)
# MetaDict = tp.Mapping[str, tp.Union[np.ndarray, torch.Tensor]]
MetaDict = tp.Mapping[str, np.ndarray]


@dataclasses.dataclass
class GoalTD3AgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.goal_td3.GoalTD3Agent"
    name: str = "goal_td3"
    reward_free: bool = omegaconf.II("reward_free")
    custom_reward: tp.Optional[str] = omegaconf.II("custom_reward")
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    update_every_steps: float = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING
    hidden_dim: int = 1024
    feature_dim: int = 512
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"
    stddev_clip: float = 0.3  # 1.0
    nstep: int = 1
    batch_size: int = 1024  # 256 for pixels
    init_critic: bool = True
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    fb_reward: bool = False
    future_ratio: float = 0
    preprocess: bool = False
    add_trunk: bool = False
    supervised: bool = True


cs = ConfigStore.instance()
cs.store(group="agent", name="goal_td3", node=GoalTD3AgentConfig)


class Critic(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)

        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2


class GoalTD3Agent:
    # pylint: disable=unused-argument

    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = GoalTD3AgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        self.goal_dim = 0
        if cfg.goal_space is not None:
            if cfg.goal_space == "quad_pos_speed":
                self.goal_dim = 7  # ugly hack
            else:
                g = next(iter(_goals.goals.funcs[cfg.goal_space].values()))()
                assert len(g.shape) == 1
                self.goal_dim = len(g)

        if self.cfg.fb_reward:
            # FB
            pt = Path("/checkpoint/atouati/ca/2022-09-09_proto_maze/results/fb_ddpg_5e-05/9/models/snapshot_1000000.pt")
            # pt = Path("/private/home/atouati/controllable_agent/url_benchmark/exp_sweep/2022.08.03/"
            #           "161531_fb_ddpg_point_mass_maze_reach_top_right_offline/1/models/snapshot_1000000.pt")
            # Contr
            # pt = Path("/private/home/atouati/controllable_agent/url_benchmark/exp_paper/"
            #           "2022.08.22_point_mass_maze_reach_top_right/100239_sf_contrastive/0/models/snapshot_1000000.pt")
            # Lap
            # pt = Path("/private/home/atouati/controllable_agent/url_benchmark/exp_paper/"
            #           "2022.08.23_point_mass_maze_reach_top_right/072210_sf_lap/1/models/snapshot_1000000.pt")
            # pt = Path("/private/home/atouati/controllable_agent/url_benchmark/exp_paper/"
            #           "2022.08.25_point_mass_maze_reach_top_right/161812_new_aps/0/models/snapshot_2000000.pt")
            print(f"loading {pt.resolve()}")
            with pt.open("rb") as f:
                payload = torch.load(f)
            fb_agent = payload["agent"]
            if hasattr(fb_agent, "feature_learner"):
                self.feature_net = fb_agent.feature_learner.feature_net
            else:
                self.feature_net = fb_agent.backward_net
            self.feature_net.eval()
            self.goal_dim = fb_agent.cfg.z_dim
            if "replay_loader" in payload.keys():
                self.precompute_cov(payload["replay_loader"])

        self.actor = Actor(self.obs_dim, self.goal_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim,
                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)

        self.critic: nn.Module = Critic(self.obs_dim, self.goal_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim,
                                        preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.critic_target: nn.Module = Critic(self.obs_dim, self.goal_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim,
                                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.reward_model: tp.Optional[torch.nn.Module] = None
        self.reward_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.reward_free:
            self.reward_model = mlp(self.obs_dim, cfg.hidden_dim, "ntanh", cfg.hidden_dim,  # type: ignore
                                    "relu", cfg.hidden_dim, "relu", 1).to(cfg.device)  # type: ignore
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def precompute_cov(self, replay_loader: ReplayBuffer) -> None:
        if not self.cfg.fb_reward:
            return None
        logger.info("computing Cov of phi to be used at inference")
        obs_list: tp.List[torch.Tensor] = []
        batch_size = 0
        while batch_size < 100000:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)  # type: ignore
            batch_size += batch.next_obs.size(0)
        obs = torch.cat(obs_list, 0)

        with torch.no_grad():
            phi = self.feature_net(obs)
        cov = torch.matmul(phi.T, phi) / phi.shape[0]
        self.inv_cov = torch.linalg.pinv(cov)

    def init_from(self, other) -> None:
        # copy parameters over
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.critic, self.critic)

    def init_meta(self, custom_reward: tp.Optional[_goals.BaseReward] = None) -> MetaDict:
        if isinstance(custom_reward, _goals.MazeMultiGoal):
            idx = np.random.choice(len(custom_reward.goals))
            desired_goal = custom_reward.goals[idx]
            meta = OrderedDict()
            meta["g"] = desired_goal
            return meta
        else:
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

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        meta = OrderedDict()
        meta['g'] = goal_array
        return meta

    def act(self, obs, meta, step, eval_mode) -> np.ndarray:
        device = torch.device(self.cfg.device)
        obs = torch.as_tensor(obs, device=device).unsqueeze(0)
        goals = []
        for value in meta.values():
            value = torch.as_tensor(value, device=device).unsqueeze(0)
            if self.cfg.fb_reward:
                with torch.no_grad():
                    goals.append(torch.matmul(self.feature_net(value), self.inv_cov))
            else:
                goals.append(value)
        goal = torch.cat(goals, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, goal, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def train_reward(self, replay_loader: ReplayBuffer) -> None:
        obs_list, reward_list = [], []
        batch_size = 0
        num_inference_steps = 10000
        while batch_size < num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            obs, action, reward, discount, next_obs = batch.to(self.cfg.device).unpack()
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
            obs, action, reward, discount, next_obs = batch.to(self.cfg.device).unpack()
            del obs, action, discount
            obs_list.append(next_obs)
            reward_list.append(reward)
            batch_size += next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[: num_inference_steps], reward[: num_inference_steps]
        test_loss = (self.reward_model(obs) - reward).pow(2).mean()
        print(f"Test Loss: {test_loss.item()}")

    @tp.no_type_check  # TODO remove
    def update_critic(self,
                      obs: torch.Tensor,
                      goal: torch.Tensor,
                      action: torch.Tensor,
                      reward: torch.Tensor,
                      discount: torch.Tensor,
                      next_obs: torch.Tensor,
                      step: int) -> tp.Dict[str, float]:
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, goal, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['stdev'] = stddev

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    @tp.no_type_check  # TODO remove
    def update_actor(self,
                     obs: torch.Tensor,
                     goal: torch.Tensor,
                     step: int) -> tp.Dict[str, float]:
        metrics = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, goal, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, goal, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_loader: ReplayBuffer, step: int,
               custom_reward: tp.Optional[_goals.BaseReward] = None) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        #import ipdb; ipdb.set_trace()

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        achieved_goal = batch.next_goal
        future_goal = batch.future_obs
        if self.cfg.goal_space:
            future_goal = batch.future_goal

        obs = batch.obs
        action = batch.action
        discount = batch.discount
        reward = batch.reward
        next_obs = batch.next_obs

        if self.cfg.reward_free:
            del reward
            assert self.reward_model is not None
            reward = self.reward_model(next_obs)

        desired_goal: torch.Tensor = torch.tensor([], dtype=torch.float32, device=self.cfg.device)
        device = torch.device(self.cfg.device)
        if isinstance(custom_reward, _goals.MazeMultiGoal):
            del reward
            if self.cfg.supervised:
                # sample uniform goal
                idx = np.random.choice(len(custom_reward.goals), size=self.cfg.batch_size)
                desired_goal = custom_reward.goals[idx]
                # convert to tensor
                desired_goal = torch.as_tensor(desired_goal, device=device)
            else:
                # sample goal from replay
                new_batch = replay_loader.sample(self.cfg.batch_size)
                new_batch = new_batch.to(self.cfg.device)
                desired_goal = new_batch.next_goal  # type: ignore
                # perm = torch.randperm(self.cfg.batch_size)
                # desired_goal = achieved_goal[perm]

            if self.cfg.future_ratio > 0:
                assert future_goal is not None
                future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)[0]
                desired_goal[future_idxs] = future_goal[future_idxs]  # type: ignore

            if self.cfg.fb_reward:
                # reward = (self.feature_net(achieved_goals) *
                #           torch.matmul(self.feature_net(desired_goals), self.inv_cov)).sum(dim=1, keepdims=True)
                with torch.no_grad():
                    desired_goal = torch.matmul(self.feature_net(desired_goal), self.inv_cov)
                    reward = (self.feature_net(achieved_goal) * desired_goal).sum(dim=1, keepdims=True)
            else:
                reward, _ = custom_reward.from_goal(achieved_goal.cpu().numpy(), desired_goal.cpu().numpy())  # type: ignore
                reward = torch.as_tensor(reward, device=device).unsqueeze(1)  # type: ignore
            # # augment obs
            # obs = torch.cat([obs, desired_goal], dim=-1)
            # next_obs = torch.cat([next_obs, desired_goal], dim=-1)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs=obs, goal=desired_goal, action=action, reward=reward,
                               discount=discount, next_obs=next_obs, step=step))

        # update actor
        metrics.update(self.update_actor(obs=obs, goal=desired_goal, step=step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.cfg.critic_target_tau)

        return metrics
