from .ddpg import *
import typing as tp
from typing import Any, Dict
import url_benchmark.goals as _goals

class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 obs_shape,
                 obs_type,
                 clip_val=5.) -> None:
        super().__init__()
        self.clip_val = clip_val

        if obs_type == "pixels":
            self.normalize_obs: nn.Module = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error


class RNDAgent(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_type = obs_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg.critic_target_tau = self.cfg.fb_target_tau
        self.rnd_rep_dim: int = 512
        self.rnd_scale: float = 1.0
        goal_dim = obs_dim
        if self.cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(self.cfg.goal_space)

        self.rnd = RND(goal_dim, self.cfg.hidden_dim, self.rnd_rep_dim, (goal_dim, ),
                       self.obs_type).to(self.cfg.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.cfg.device)

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
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.cfg.lr)

        self.train()
        self.critic_target.train()
        self.rnd.train()


    def train(self, training: bool = True) -> None:
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    # pylint: disable=unused-argument
    def update_rnd(self, obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}
        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['rnd_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, step) -> Any:
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward


    def act(self, obs, step, eval_mode, exploratory_action = False) -> np.ndarray:
        obs = torch.as_tensor(obs, device=self.cfg.device).unsqueeze(0)
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.cfg.num_expl_steps or exploratory_action == True:
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

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # print("In RND update step", step)

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        metrics.update(self.update_rnd(next_goal, step))
        with torch.no_grad():
            intr_reward = self.compute_intr_reward(next_goal, step)
            reward = intr_reward

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['intr_reward'] = intr_reward.mean().item()
            metrics['intr_reward_std'] = intr_reward.std().item()
            metrics['pred_error_mean'] = self.intrinsic_reward_rms.M.item()
            metrics['pred_error_std'] = torch.sqrt(self.intrinsic_reward_rms.S).item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.cfg.critic_target_tau)

        return metrics
