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
    # mode
    reward_free: bool = True
    # train settings
    num_train_episodes: int = 2000
    # snapshot
    eval_every_episodes: int = 10
    load_replay_buffer: tp.Optional[str] = None
    # replay buffer
    # replay_buffer_num_workers: int = 4
    # nstep: int = omgcf.II("agent.nstep")
    # misc
    save_train_video: bool = False
    update_replay_buffer: bool = True
    num_rollout_episodes: int = 10
    num_agent_updates: int = 10


ConfigStore.instance().store(name="workspace_config", node=OnlinetrainConfig)


class Workspace(pretrain.BaseWorkspace[OnlinetrainConfig]):
    def __init__(self, cfg: OnlinetrainConfig) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        self._last_processed_step = 0  # for checkpointing
        if not cfg.update_replay_buffer:
            cfg.num_seed_frames = -1
            if cfg.load_replay_buffer is None:
                raise ValueError("If update_replay_buffer is False, load_replay_buffer must be provided")
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])

    def _play_episode(self, log_metrics: bool = True) -> None:
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        episode_step = 0
        episode_reward = 0.0
        z_correl = 0.0
        physics_agg = dmc.PhysicsAggregator()
        custom_reward = self._make_custom_reward(seed=self.global_step)
        while not time_step.last():
            meta = self.agent.update_meta(meta, self.global_step, time_step, replay_loader=self.replay_loader)
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            time_step = self.train_env.step(action)
            if custom_reward is not None:
                time_step.reward = custom_reward.from_env(self.train_env)
            physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            if isinstance(self.agent, agents.FBDDPGAgent):
                z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1
        # log episode stats
        if log_metrics:
            self.train_video_recorder.save(f'{self.global_frame}.mp4')
            elapsed_time, total_time = self.timer.reset()
            episode_frame = episode_step * self.cfg.action_repeat
            with self.logger.log_and_dump_ctx(self.global_frame,
                                              ty='train') as log:
                log('fps', episode_frame / elapsed_time)
                log('z_correl', z_correl)
                log('total_time', total_time)
                log('episode_reward', episode_reward)
                log('episode_length', episode_frame)
                log('episode', self.global_episode)
                log('buffer_size', len(self.replay_loader))
                log('step', self.global_step)
                for key, val in physics_agg.dump():
                    log(key, val)

    def _checkpoint_if_need_be(self) -> None:
        # save checkpoint to reload
        if self.global_step // self.cfg.checkpoint_every != self._last_processed_step // self.cfg.checkpoint_every:
            self.save_checkpoint(self._checkpoint_filepath)
        if any(self._last_processed_step < x <= self.global_step for x in self.cfg.snapshot_at):
            self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
        self._last_processed_step = self.global_step

    def train(self) -> None:
        metrics: tp.Optional[tp.Dict[str, float]] = None
        while self.global_episode < self.cfg.num_train_episodes:
            logger.info(f"rollout {self.cfg.num_rollout_episodes} episodes...")
            for _ in range(self.cfg.num_rollout_episodes):
                self._play_episode(log_metrics=metrics is not None)  # logging requires all metrics available
                self.global_episode += 1
            # update the agent
            if self.global_frame > self.cfg.num_seed_frames:
                # TODO: reward_free should be handled in the agent update itself !
                # replay = (x.with_no_reward() if self.cfg.reward_free else x for x in self.replay_loader)
                logger.info(f"Agent update for {self.cfg.num_agent_updates}...")
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(self.replay_loader, self.global_step)
                if metrics is not None:
                    self.logger.log_metrics(metrics, self.global_step, ty='train')
            # evaluate
            if not self.global_episode % self.cfg.eval_every_episodes:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
            if self.cfg.use_hiplog and self.logger.hiplog.content:
                self.logger.hiplog.write()  # write to hiplog only once per episode
            # checkpoint
            self._checkpoint_if_need_be()
        self.save_checkpoint(self._checkpoint_filepath)
        self.finalize()


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
