# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp

from url_benchmark import pretrain  # NEEDS TO BE FIRST NON-STANDARD IMPORT (sets up env variables)

import omegaconf as omgcf
import hydra
from hydra.core.config_store import ConfigStore
import torch

from url_benchmark import dmc
from url_benchmark import utils
from url_benchmark import agent as agents
from url_benchmark.video import TrainVideoRecorder
from url_benchmark.goals import WalkerYogaReward
from collections import defaultdict

import numpy as np
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import url_benchmark
# we need to add url_benchmarl to be able to reload legacy checkpoints
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))


@dataclasses.dataclass
class OnlinetrainConfig(pretrain.Config):
    reward_free: bool = True
    # train settings
    num_train_episodes: int = 2000
    num_seed_episodes: int = 10
    # snapshot
    eval_every_episodes: int = 1
    load_replay_buffer: tp.Optional[str] = None
    # replay buffer
    # replay_buffer_num_workers: int = 4
    # nstep: int = omgcf.II("agent.nstep")
    # misc
    save_train_video: bool = False
    update_replay_buffer: bool = True
    num_rollout_episodes: int = 5 # num of episodes for current z to play traj for to update replay buffer
    num_agent_updates: int = 10
    update_target_net_prob: float = 0.1 # 20% of model updates will lead to target updates
    exploratory_diffusion_after_every_episode: int = 10 # adds uniform exploration every 10 episodes
    exploration_in_last_steps: float=0.8



ConfigStore.instance().store(name="workspace_config", node=OnlinetrainConfig)

class Workspace(pretrain.BaseWorkspace[OnlinetrainConfig]):
    def __init__(self, cfg: OnlinetrainConfig) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id,
                                                       use_wandb=self.cfg.use_wandb)
        self._last_processed_step = 0  # for checkpointing
        self.z_list = []
        if not cfg.update_replay_buffer:
            cfg.num_seed_frames = -1
            if cfg.load_replay_buffer is None:
                raise ValueError("If update_replay_buffer is False, load_replay_buffer must be provided")
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])

    def _z_rank(self)-> tp.Dict[str, float]:
        # compute the rank of z cov matrix to check the dissimilarity in z generated
        metrics: tp.Dict[str, float] = {}
        z_mat = np.stack(self.z_list, axis=0)
        z_cov = np.matmul(z_mat, z_mat.T)
        rank = np.linalg.matrix_rank(z_cov)
        metrics["z_rank"] = rank
        metrics["z_num"] = z_mat.shape[0]
        metrics["z_episode"] = self.global_episode
        return metrics


    # def add_noise_to_state(self, time_step) -> tp.Any:
    #     # adding normally distributed noise to state vector to test robustness of RND and FB + exploration
    #     s = time_step.


    def _play_episode_with_exploratory_agent(self) -> None:
        metrics = None
        # [RNDAgent(FB representation agnostic) vs ExploratoryAgent (guided by FB representation)]
        sampling_agent = self.rnd_exploratory_agent if self.cfg.explore_rnd else self.exploratory_agent
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_loader.add(time_step, meta)
        # self.train_video_recorder.init(time_step.observation)
        episode_step = 0
        episode_reward = 0.0
        exploratory_action = False
        # z_correl = 0.0
        # physics_agg = dmc.PhysicsAggregator()
        # custom_reward = self._make_custom_reward(seed=self.global_step)

        while not time_step.last():
            if self.global_episode > self.cfg.num_seed_episodes_explore:
                if not self.global_step % self.cfg.update_cov_steps and not self.cfg.intr_reward_FBloss and not self.cfg.explore_rnd:
                    # update cov B matrix
                    icov_F, icov_B = self.agent.get_icov_FB(self.replay_loader, self.cfg.exp_cov_F, self.cfg.exp_cov_B, self.cfg.exp_cov_FB)
                    self.exploratory_agent.set_icov_matrix(icov_matrix_F = icov_F, icov_matrix_B=icov_B)
                if not self.global_step % self.cfg.update_exploratory_agent_every_steps:
                    metrics = sampling_agent.update(self.replay_loader, self.global_step)
                    metrics["episode"] = self.global_episode
                    self.logger.log_metrics(metrics, self.global_step, ty='train') # logging metrics after every update
                if self.cfg.uniform_explore_after_few_episodes and self.global_episode % self.cfg.exploratory_diffusion_after_every_episode == 0:
                    if episode_step >= self.replay_loader.episode_length * self.cfg.exploration_in_last_steps: # last 100 steps, add uniform exploration for diffusion
                        exploratory_action = True  # adds uniform action for 5 episodes every 500 episodes for adding new samples to replay buffer
            with torch.no_grad(), utils.eval_mode(sampling_agent):
                action = sampling_agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False,
                                        exploratory_action = exploratory_action)

            time_step = self.train_env.step(action)
            # if custom_reward is not None:
            #     time_step.reward = custom_reward.from_env(self.train_env)
            # physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta)
            # self.train_video_recorder.record(time_step.observation)
            # if isinstance(self.agent, agents.FBDDPGAgent):
            #     z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1

        # self.train_video_recorder.save(f'{self.global_frame}.mp4')
        elapsed_time, total_time = self.timer.reset()
        episode_frame = episode_step * self.cfg.action_repeat
        if metrics is not None:
            with self.logger.log_and_dump_ctx(self.global_frame,
                                              ty='train') as log:
                log('fps', episode_frame / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', episode_reward)
                log('episode_length', episode_frame)
                log('episode', self.global_episode)
                log('buffer_size', len(self.replay_loader))
                log('step', self.global_step)
                # for key, val in physics_agg.dump():
                #     log(key, val)
        if self.cfg.use_hiplog and self.logger.hiplog.content:
            self.logger.hiplog.write()

    def _play_episode(self, log_metrics: bool = True) -> None:
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        episode_step = 0
        episode_reward = 0.0
        exploratory_action = False
        # z_correl = 0.0
        # physics_agg = dmc.PhysicsAggregator()
        # custom_reward = self._make_custom_reward(seed=self.global_step)

        while not time_step.last():
            # generating z only for exploration, not for saving z in replay buffer
            if self.cfg.explore and isinstance(self.agent, agents.FBDDPGAgent) and self.global_episode > self.cfg.num_seed_episodes_explore:
                if not self.cfg.set_epsilon_exploration:
                    self.agent.set_exploration(explore=True) # always explore, no epsilon exploration
                elif np.random.rand() < self.cfg.epsilon_explore_val: # set_epsilon_exploration is True
                    self.agent.set_exploration(explore=True) # With epsilon, explore=True
                else:
                    self.agent.set_exploration(explore=False) # With 1-epsilon, explore=False
            meta = self.agent.update_meta(meta, self.global_step, time_step, replay_loader=self.replay_loader)
            if self.cfg.uniform_explore_after_few_episodes and self.global_episode % self.cfg.exploratory_diffusion_after_every_episode == 0:
                if episode_step >= self.replay_loader.episode_length * self.cfg.exploration_in_last_steps:  # last 100 steps, add uniform exploration for diffusion
                    exploratory_action = True  # adds uniform action for 5 episodes every 500 episodes for adding new samples to replay buffer
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False,
                                        exploratory_action = exploratory_action)

            time_step = self.train_env.step(action)
            # if custom_reward is not None:
            #     time_step.reward = custom_reward.from_env(self.train_env)
            # physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta)
            # self.train_video_recorder.record(time_step.observation)
            # if isinstance(self.agent, agents.FBDDPGAgent):
            #     z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1

        # log episode stats
        if log_metrics:
            # self.train_video_recorder.save(f'{self.global_frame}.mp4')
            elapsed_time, total_time = self.timer.reset()
            episode_frame = episode_step * self.cfg.action_repeat
            with self.logger.log_and_dump_ctx(self.global_frame,
                                              ty='train') as log:
                log('fps', episode_frame / elapsed_time)
                # log('z_correl', z_correl)
                log('total_time', total_time)
                log('episode_reward', episode_reward)
                log('episode_length', episode_frame)
                log('episode', self.global_episode)
                log('buffer_size', len(self.replay_loader))
                log('step', self.global_step)
                # for key, val in physics_agg.dump():
                #     log(key, val)
        if self.cfg.use_hiplog and self.logger.hiplog.content:
            self.logger.hiplog.write()

    def _checkpoint_if_need_be(self) -> None:
        # save checkpoint to reload
        if self.global_step // self.cfg.checkpoint_every != self._last_processed_step // self.cfg.checkpoint_every:
            self.save_checkpoint(self._checkpoint_filepath)
        if any(self._last_processed_step < x <= self.global_step for x in self.cfg.snapshot_at):
            self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
        self._last_processed_step = self.global_step

    def eval_walker_goals(self) -> None:
        reward_cls = WalkerYogaReward()
        total_success = defaultdict(list)
        for task in reward_cls._goals.keys():
            meta = self.agent.get_goal_meta(reward_cls.target_obs[task])
            for episode in range(self.cfg.num_eval_episodes):
                # self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                time_step = self.eval_env.reset()
                success = 0.0
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                0,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    success += int(abs(reward_cls.compute_reward(time_step.physics, task)) < 0.7)
                    # self.video_recorder.record(self.eval_env)
                # last_reward = reward_cls.compute_reward(time_step.physics, task)
                # success = int(abs(last_reward) < 0.7)
                # self.video_recorder.save(f'{task}_{self.global_frame}.mp4')
                # total_reward[task].append(abs(last_reward))
                total_success[task].append(success)
                # print("completed ", task, " with reward", total_reward[task])
        avg_success = float(np.mean([total_success[task] for task in reward_cls._goals.keys()]))
        # avg_reward = float(np.mean([total_reward[task] for task in reward_cls._goals.keys()]))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            for task in reward_cls._goals.keys():
                # log(f'{task}_episode_reward', float(np.mean(total_reward[task])))
                log(f'yoga_{task}_success', float(np.mean(total_success[task])))
            log(f'avg_yoga_success', float(np.mean(avg_success)))
            # log(f'avg_tasks_episode_reward', float(np.mean(avg_reward)))
            log(f'step', self.global_step)
            log(f'episode', self.global_episode)


    def train(self) -> None:
        metrics = None
        # have a separate agent network which builds the intrinsic reward
        while self.global_episode < self.cfg.num_train_episodes:
            logger.info(f"rollout {self.cfg.num_rollout_episodes} episodes...")
            # collect dataset
            for _ in range(self.cfg.num_rollout_episodes):
                if self.cfg.explore and (self.cfg.separate_exploratory_agent or self.cfg.explore_rnd):
                    self._play_episode_with_exploratory_agent()
                else:
                    self._play_episode(log_metrics=metrics is not None)  # logging requires all metrics available
                self.global_episode += 1

            # update the agent
            if self.global_episode > self.cfg.num_seed_episodes: # for seed episodes would just collect data and not update network
                # num_agent_updates = int(self.cfg.num_agent_updates*(1 +  float(4 * self.global_episode)/self.cfg.num_train_episodes))
                num_agent_updates = self.cfg.num_agent_updates # constant updates
                for agent_update_ind in range(num_agent_updates -1):
                    self.agent.update(self.replay_loader, self.global_step)
                    if not agent_update_ind % int(1/self.cfg.update_target_net_prob): # update the target net slowly for stability
                        self.agent.update_target_net()
                metrics = self.agent.update(self.replay_loader, self.global_step)
                metrics["episode"] = self.global_episode
                self.agent.update_target_net()
                self.logger.log_metrics(metrics, self.global_step, ty='train')
            # if metrics is not None:
            #     with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
            #         log('episode', self.global_episode)

            # evaluate
            if not self.global_episode % self.cfg.eval_every_episodes:
                self.logger.log('eval_total_time', self.timer.total_time(),self.global_frame)
                if self.cfg.custom_reward_maze == "maze_multi_goal":
                    self.eval_maze_goals()
                    # print("eval multi goals")
                elif self.cfg.custom_reward_walker == "walker_yoga":
                    self.eval_walker_goals() # eval yoga walker goals
                    self.eval() # eval original task goals
                elif self.cfg.custom_reward_mujoco == "mujoco_tasks": # eval - > run, walk, flip, stand
                    # print("in evaluation")
                    self.eval_mlt_tasks()
                else:
                    self.eval()
            if self.cfg.use_hiplog and self.logger.hiplog.content:
                self.logger.hiplog.write()  # write to hiplog only once per episode
            # checkpointing
            if not self.global_episode % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_episode}.pt'))
        self.save_checkpoint(self._checkpoint_filepath)
        self.finalize()


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
