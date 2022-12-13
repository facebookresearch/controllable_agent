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

from url_benchmark import goals as _goals
from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer # pylint: disable=unused-import
from url_benchmark.replay_buffer import EpisodeBatch # pylint: disable=unused-import
from url_benchmark import agent as agents

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

@dataclasses.dataclass
class OfflineConfig(pretrain.Config):
    # training
    num_grad_steps: int = 1000000
    num_seed_frames: int = 0
    log_every_steps: int = 1000
    # eval
    num_eval_episodes: int = 10
    eval_every_steps: int = 10000
    # dataset
    load_replay_buffer: tp.Optional[str] = None
    expl_agent: str = "proto"
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")  # make sure to update this if you change hydra run dir
    # misc
    experiment: str = "offline"
    reward_free: bool = False


ConfigStore.instance().store(name="workspace_config", node=OfflineConfig)


class Workspace(pretrain.BaseWorkspace[OfflineConfig]):
    def __init__(self, cfg: OfflineConfig) -> None:
        super().__init__(cfg)
        self.agent.cfg.update_every_steps = 1
        datasets_dir = self.work_dir / cfg.replay_buffer_dir
        replay_dir = datasets_dir.resolve() / self.domain / cfg.expl_agent / 'buffer'
        print(f'replay dir: {replay_dir}')

        # self.replay_loader = ReplayBuffer([],  # self._data_specs, [],  # meta_specs = []
        #                                   cfg.batch_size, cfg.replay_buffer_episodes,
        #                                   cfg.discount, True)

        if self.cfg.load_replay_buffer is not None:
            print("loading Replay from %s", self.cfg.load_replay_buffer)
            self.load_checkpoint(self.cfg.load_replay_buffer, only=["replay_loader"])
            # with open(self.cfg.load_replay_buffer, 'rb') as f:
            #     content = torch.load(f)
            #     if isinstance(content, dict):
            #         content = content["replay_loader"]
            #     # assert isinstance(content, ReplayBuffer)
            #     self.replay_loader = content
        else:
            relabeled_replay_file_path = replay_dir / f"../relabeled_replay_{cfg.task}_{cfg.replay_buffer_episodes}.pt"
            if relabeled_replay_file_path.exists():
                print("loading Replay from %s", relabeled_replay_file_path.resolve())
                self.load_checkpoint(relabeled_replay_file_path, only=["replay_loader"])
                # with relabeled_replay_file_path.open('rb') as f:
                #     self.replay_loader = torch.load(f)
            else:
                print("loading and relabeling...")
                goal_func = None if cfg.goal_space is None else _goals.goal_spaces.funcs[self.domain][cfg.goal_space]
                self.replay_loader.load(self.train_env, replay_dir, relabel=True, goal_func=goal_func)
                print("loading is done")
                with relabeled_replay_file_path.open('wb') as f:
                    torch.save(self.replay_loader, f)

        self.replay_loader._future = cfg.future
        self.replay_loader._discount = cfg.discount
        # self.replay_loader._full = True
        self.replay_loader._max_episodes = len(self.replay_loader._storage["discount"])

        if isinstance(self.agent, agents.GoalTD3Agent) and self.agent.cfg.fb_reward:
            self.agent.precompute_cov(self.replay_loader)
            

    def train(self):
        train_until_step = utils.Until(self.cfg.num_grad_steps)
        eval_every_step = utils.Every(self.cfg.eval_every_steps)
        log_every_step = utils.Every(self.cfg.log_every_steps)

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_step)
                if self.cfg.custom_reward == "maze_multi_goal":
                    self.eval_maze_goals()
                else:
                    self.eval()

            if isinstance(self.agent, agents.GoalTD3Agent):
                metrics = self.agent.update(self.replay_loader, self.global_step, self.reward_cls)
            else:
                metrics = self.agent.update(self.replay_loader, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', self.cfg.log_every_steps / elapsed_time)
                    log('total_time', total_time)
                    log('step', self.global_step)
            self.global_step += 1
            # try to save snapshot
            if self.global_frame in self.cfg.snapshot_at:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'), exclude=["replay_loader"])
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath, exclude=["replay_loader"])
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()

    # def load_checkpoint(self, fp: tp.Union[Path, str]) -> None:
    #     fp = Path(fp)
    #     with fp.open('rb') as f:
    #         payload = torch.load(f)
    #     self.agent.init_from(payload['agent'])


@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)  # type: ignore
    # for _ in range(10):
    #     workspace.eval()
    if isinstance(workspace.agent, agents.DDPGAgent):
        if workspace.agent.reward_free:
            workspace.agent.train_reward(workspace.replay_loader)
    workspace.train()


if __name__ == '__main__':
    main()
