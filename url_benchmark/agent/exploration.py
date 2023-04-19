import torch

from .ddpg import *
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from typing import Any, Dict, Tuple
import torch.nn.functional as F
from url_benchmark.utils import EMA, EMA_STD


class ExplorationAgent(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, cfg, cov_F, cov_B, cov_FB, intr_rew_FBloss, worst_z, ema_intr_reward, reward_prioritization, main_agent) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_dim =obs_dim
        self.action_dim = action_dim
        self.obs_type = obs_type
        self.cov_F = cov_F
        self.cov_B = cov_B
        self.cov_FB = cov_FB
        self.intr_rew_FBloss = intr_rew_FBloss
        self.compute_worst_z = worst_z
        self.do_ema_intr_reward_metric = ema_intr_reward
        self.reward_prioritization = reward_prioritization
        self.main_agent = main_agent
        self.cfg.name = "exploration"
        self.cfg.reward_free = False
        self.cfg.critic_target_tau = self.cfg.fb_target_tau
        self.ema_r_F = EMA_STD(self.cfg.device)
        self.ema_r_B = EMA_STD(self.cfg.device)
        self.ema_r_FBloss = EMA_STD(self.cfg.device)
        self.ema_reward_priority_FB = EMA_STD(self.cfg.device)
        self.ema_reward_priority_F = EMA_STD(self.cfg.device)
        self.ema_reward_priority_B = EMA_STD(self.cfg.device)
        self.ema_metric_FB = EMA()
        self.ema_metric_F = EMA()
        self.ema_metric_B = EMA()

        if self.cov_FB or self.cov_F or self.cov_B:
            self.set_FB_network(F = self.main_agent.forward_net, B = self.main_agent.backward_net)

        # agent models
        self.actor = Actor(self.obs_type, self.obs_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode, exploratory_action = False) -> np.ndarray:
        obs = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0)
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.cfg.num_expl_steps or exploratory_action:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs: torch.Tensor,
                      action: torch.Tensor,
                      reward: torch.Tensor,
                      discount: torch.Tensor,
                      next_obs: torch.Tensor,
                      step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['expl_q1'] = Q1.mean().item()
            metrics['expl_critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    @tp.no_type_check  # TODO remove
    def update_actor(self, obs: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['expl_actor_loss'] = actor_loss.item()
            metrics['expl_actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def set_icov_matrix(self, icov_matrix_F, icov_matrix_B):
        self.icov_matrix_F = icov_matrix_F
        self.icov_matrix_B = icov_matrix_B
        if icov_matrix_F is not None and self.do_ema_intr_reward_metric:
            self.icov_matrix_F = self.ema_metric_F(icov_matrix_F)
        if icov_matrix_B is not None and self.do_ema_intr_reward_metric:
            self.icov_matrix_B = self.ema_metric_B(icov_matrix_B)

    def set_FB_network(self, F, B):
        self.F_net = F
        self.B_net = B

    def get_cov_metrics(self, A, rank_threshold = 1e-3):
        diag = A.diag().mean.item()
        L = torch.linalg.eigvals(A).type(torch.float64)
        rank = torch.where(L.absolute()>rank_threshold, 1, 0).sum().item()
        return diag, rank

    def compute_intr_reward(self, obs, action, next_obs, discount, next_goal, z, step) -> Any:
        # compute Fb elliptical reward function
        # r = \sqrt{B^T \Sigma_B B}
        metrics: tp.Dict[str, float] = {}
        if self.intr_rew_FBloss: # compute reward as FB loss
            with torch.no_grad():
                if self.compute_worst_z:
                    reward, _ = self.main_agent.compute_worst_z_fb_loss_nograd(obs, action, discount, next_obs, next_goal, step)
                else:
                    reward = self.main_agent.compute_fb_loss_with_nograd(obs, action, discount, next_obs, next_goal, z, step)
            if self.do_ema_intr_reward_metric:
                reward = self.ema_metric_FB(reward)
            std = self.ema_r_FBloss(reward)  # normalize reward by exp moving avg std
            reward /= std
            if self.reward_prioritization:
                _ = self.ema_reward_priority_FB(reward)
                mean_rp = self.ema_reward_priority_FB.mean
                reward = torch.maximum(reward - mean_rp, torch.tensor(0))
            reward = torch.unsqueeze(reward, dim=1)
        elif self.cov_FB:
            reward_F =  self.elliptical_bonus_F(obs, action, z)
            reward_B = self.elliptical_bonus_B(next_obs)
            metrics["intr_rew_F"] = reward_F.mean().item()
            metrics["intr_rew_B"] = reward_B.mean().item()
            reward = reward_F + reward_B
        elif self.cov_F:
            reward_F = self.elliptical_bonus_F(obs, action, z)
            metrics["intr_rew_F"] = reward_F.mean().item()
            reward = reward_F
        else: # intrinsic reward as cov B
            reward_B = self.elliptical_bonus_B(next_obs)
            metrics["intr_rew_B"] = reward_B.mean().item()
            reward = reward_B

        return reward, metrics


    def elliptical_bonus_B(self, obs) -> Any:
        with torch.no_grad():
            B = self.B_net(obs)  # b X d
        reward = torch.sqrt((torch.matmul(B, self.icov_matrix_B) * B).sum(dim=1))
        std = self.ema_r_B(reward)  # normalize reward by exp moving avg std
        reward /= std
        if self.reward_prioritization:
            _ = self.ema_reward_priority_B(reward)
            mean_rp = self.ema_reward_priority_B.mean
            reward = torch.maximum(reward - mean_rp, torch.tensor(0))
        reward = torch.unsqueeze(reward, dim=1)
        return reward

    def elliptical_bonus_F(self, obs, action, z) -> Any:
        with torch.no_grad():
            F1, F2 = self.F_net(obs, z, action)  # [s x d]
        reward = 0.5 * sum(torch.sqrt((torch.matmul(F, self.icov_matrix_F) * F).sum(dim=1)) for F in [F1, F2])
        std = self.ema_r_F(reward)
        reward /= std
        if self.reward_prioritization:
            _ = self.ema_reward_priority_F(reward)
            mean_rp = self.ema_reward_priority_F.mean
            reward = torch.maximum(reward - mean_rp, torch.tensor(0))
        reward = torch.unsqueeze(reward, dim=1)
        return reward


    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        z = self.main_agent.sample_z(self.cfg.batch_size, device=self.cfg.device)
        reward, metric_reward = self.compute_intr_reward(obs, action, next_obs, discount, next_goal, z, step)

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['intr_reward'] = reward.mean().item()
            metrics['intr_reward_std'] = reward.std().item()
            metrics.update(metric_reward)
        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.cfg.critic_target_tau)

        return metrics