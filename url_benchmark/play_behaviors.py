# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import dataclasses
import typing as tp

import hydra
from hydra.core.config_store import ConfigStore
import torch
import omegaconf as omgcf


from url_benchmark.pretrain import make_agent
from url_benchmark import dmc
from url_benchmark import utils
from url_benchmark.video import VideoRecorder
from url_benchmark import agent as agents
from url_benchmark import goals as _goals
from typing import Any

torch.backends.cudnn.benchmark = True


# # # Config # # #

@dataclasses.dataclass
class PlayConfig:
    agent: tp.Any
    # mode
    reward_free: bool = True
    # task settings
    task: str = "walker_stand"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    action_repeat: int = 1  # set to 2 for pixels
    discount: float = 0.99
    goal_space: str = "simplified"
    # train settings
    num_train_frames: int = 100010
    num_seed_frames: int = 0
    # eval
    eval_every_frames: int = 10000
    num_eval_episodes: int = 10
    # snapshot
    snapshot_ts: int = 2000000
    snapshot_base_dir: str = omgcf.SI("./pretrained_models")
    # replay buffer
    replay_buffer_size: int = 1000000
    replay_buffer_num_workers: int = 4
    batch_size: int = omgcf.II("agent.batch_size")
    nstep: int = omgcf.II("agent.nstep")
    update_encoder: bool = False  # should always be true for pre-training
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = True
    save_train_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    use_hiplog: bool = False
    # experiment
    experiment: str = "exp"


# loaded as base_finetune in finetune.yaml
# we keep the yaml since it's easier to configure plugins from it
ConfigStore.instance().store(name="workspace_config", node=PlayConfig)

# # # Implem # # #


class Workspace:
    def __init__(self, cfg: PlayConfig) -> None:
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create envs

        self.env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                            cfg.action_repeat, cfg.seed, cfg.goal_space)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.env.observation_spec(),
                                self.env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

    def play(self) -> None:
        episode, total_reward = 0, 0.0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            total_reward = 0
            if isinstance(self.agent, agents.FBDDPGAgent):
                g = _goals.goals.funcs[self.cfg.goal_space][self.cfg.task]()
                meta = self.agent.get_goal_meta(g)
            else:
                meta = self.agent.init_meta()
            time_step = self.env.reset()
            self.video_recorder.init(self.env)
            step = 0
            eval_until_step = utils.Until(1000)
            while eval_until_step(step):
                # print(f'episode {episode}, step {step}')
                # while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            1,
                                            eval_mode=True)
                time_step = self.env.step(action)
                self.video_recorder.record(self.env)
                total_reward += time_step.reward
                # print(time_step.goal[2])
                step += 1

            episode += 1
            print(total_reward)
            self.video_recorder.save(f'{episode}.mp4')

    def load_snapshot(self) -> Any:
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        # domain, _ = self.cfg.task.split('_', 1)
        # snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        snapshot_dir = snapshot_base_dir

        def try_load():
            # snapshot = snapshot_dir / str(
            #     seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            snapshot = snapshot_dir / f'snapshot_{self.cfg.snapshot_ts}.pt'
            # if not snapshot.exists():
            #     return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load()
        return payload


@hydra.main(config_path='.', config_name='base_config')
def main(cfg) -> None:
    workspace = Workspace(cfg)
    workspace.play()


if __name__ == '__main__':
    main()
