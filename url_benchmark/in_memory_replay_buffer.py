# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import logging
import typing as tp
import dataclasses
import collections
from pathlib import Path

import numpy as np
import torch
from dm_env import specs, TimeStep
from tqdm import tqdm
from url_benchmark.replay_buffer import EpisodeBatch
from url_benchmark.dmc import ExtendedGoalTimeStep

Specs = tp.Sequence[specs.Array]
logger = logging.getLogger(__name__)

EpisodeTuple = tp.Tuple[np.ndarray, ...]
Episode = tp.Dict[str, np.ndarray]
T = tp.TypeVar("T", np.ndarray, torch.Tensor)


def episode_len(episode: Episode) -> int:
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def load_episode(fn: Path) -> tp.Dict[str, np.ndarray]:
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    return episode  # type: ignore


def relabel_episode(env: tp.Any, episode: tp.Dict[str, np.ndarray], goal_func: tp.Any) -> tp.Dict[str, np.ndarray]:
    goals = []
    rewards = []
    states = episode['physics']
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full((1,), reward, dtype=np.float32)
        rewards.append(reward)
        if goal_func is not None:
            goals.append(goal_func(env))
    episode['reward'] = np.array(rewards, dtype=np.float32)
    if goals:
        episode['goal'] = np.array(goals, dtype=np.float32)
    return episode

# class ReplayBufferIterable:
#     def __init__(self, replay_buffer: "ReplayBuffer") -> None:
#         self._replay_buffer = replay_buffer
#
#     def __next__(self) -> EpisodeBatch:
#         return self._replay_buffer.sample()


class ReplayBuffer:
    def __init__(self,
                 max_episodes: int, discount: float, future: float) -> None:
        # data_specs: Specs,
        # self._data_specs = tuple(data_specs)
        # self._meta_specs = tuple(meta_specs)
        # self._batch_size = batch_size
        self._max_episodes = max_episodes
        self._discount = discount
        assert 0 <= future <= 1
        self._future = future
        self._current_episode: tp.Dict[str, tp.List[np.ndarray]] = collections.defaultdict(list)
        self._idx = 0
        self._full = False
        self._num_transitions = 0
        self._storage: tp.Dict[str, np.ndarray] = collections.defaultdict()
        self._collected_episodes = 0
        self._batch_names = set(field.name for field in dataclasses.fields(ExtendedGoalTimeStep))

    def __len__(self) -> int:
        return self._max_episodes if self._full else self._idx

    def add(self, time_step: TimeStep, meta: tp.Mapping[str, np.ndarray]) -> None:
        dtype = np.float32
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for field in dataclasses.fields(time_step):
            value = time_step[field.name]
            if np.isscalar(value):
                value = np.full((1,), value, dtype=dtype)
            if isinstance(value, np.ndarray):
                self._current_episode[field.name].append(np.array(value, dtype=dtype))
        if time_step.last():
            if not hasattr(self, "_batch_names"):
                self._batch_names = set(field.name for field in dataclasses.fields(ExtendedGoalTimeStep))
            for name, value_list in self._current_episode.items():
                values = np.array(value_list, dtype)
                if name in self._storage and name not in self._batch_names:
                    if values.shape != self._storage[name].shape[1:]:
                        logger.warning(f"Reinitializing meta data {name} to 0 since it has changed shapes.")
                        del self._storage[name]
                if name not in self._storage:
                    # first iteration, the buffer is created with appropriate size
                    self._storage[name] = np.empty((self._max_episodes,) + values.shape, dtype=dtype)
                self._storage[name][self._idx] = values
            self._current_episode = collections.defaultdict(list)
            self._collected_episodes += 1
            self._idx = (self._idx + 1) % self._max_episodes
            self._full = self._full or self._idx == 0

    @property
    def episode_length(self) -> int:
        data = next(iter(self._storage.values()))  # should be all equal (caution: this is not enforced)
        return int(data.shape[1] - 1)

    def sample(self, batch_size, custom_reward: tp.Optional[tp.Any] = None, with_physics: bool = False) -> EpisodeBatch:
        if not hasattr(self, "_batch_names"):
            self._batch_names = set(field.name for field in dataclasses.fields(ExtendedGoalTimeStep))
        if not isinstance(self._future, float):
            assert isinstance(self._future, bool)
            self._future = float(self._future)
        ep_idx = np.random.randint(0, len(self), size=batch_size)
        # add +1 for the first dummy transition
        step_idx = np.random.randint(0, self.episode_length, size=batch_size) + 1
        if self._future < 1:
            # future_idx = step_idx + np.random.randint(0, self.episode_length - step_idx + 1, size=self._batch_size)
            future_idx = step_idx + np.random.geometric(p=(1 - self._future), size=batch_size)
            future_idx = np.clip(future_idx, 0, self.episode_length)
        meta = {name: data[ep_idx, step_idx - 1] for name, data in self._storage.items() if name not in self._batch_names}
        obs = self._storage['observation'][ep_idx, step_idx - 1]
        action = self._storage['action'][ep_idx, step_idx]
        next_obs = self._storage['observation'][ep_idx, step_idx]
        phy = self._storage['physics'][ep_idx, step_idx]
        if custom_reward is not None:
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
        else:
            reward = self._storage['reward'][ep_idx, step_idx]
        discount = self._discount * self._storage['discount'][ep_idx, step_idx]
        goal: tp.Optional[np.ndarray] = None
        next_goal: tp.Optional[np.ndarray] = None
        future_obs: tp.Optional[np.ndarray] = None
        future_goal: tp.Optional[np.ndarray] = None
        if 'goal' in self._storage.keys():
            goal = self._storage['goal'][ep_idx, step_idx - 1]
            next_goal = self._storage['goal'][ep_idx, step_idx]
            if self._future < 1:
                future_goal = self._storage['goal'][ep_idx, future_idx - 1]
        # elif self._future:
        if self._future < 1:
            future_obs = self._storage['observation'][ep_idx, future_idx - 1]
        additional = {}
        if with_physics:
            additional["_physics"] = phy
        # TODO remove type ignore when working
        return EpisodeBatch(obs=obs, goal=goal, action=action, reward=reward, discount=discount,
                            next_obs=next_obs, next_goal=next_goal,
                            future_obs=future_obs, future_goal=future_goal, meta=meta, **additional)

    def load(self, env: tp.Any, replay_dir: Path, relabel: bool = True, goal_func: tp.Any = None) -> None:
        eps_fns = sorted(replay_dir.glob('*.npz'))
        for eps_fn in tqdm(eps_fns):
            if self._full:
                break
            episode = load_episode(eps_fn)
            if relabel:
                episode = relabel_episode(env, episode, goal_func)
            # for field in dataclasses.fields(TimeStep):
            for name, values in episode.items():
                # values = episode[field.name]
                if name not in self._storage:
                    # first iteration, the buffer is created with appropriate size
                    self._storage[name] = np.empty((self._max_episodes,) + values.shape, dtype=np.float32)
                self._storage[name][self._idx] = np.array(values, dtype=np.float32)
            self._idx = (self._idx + 1) % self._max_episodes
            self._full = self._full or self._idx == 0

    def relabel(self, custom_reward) -> None:

        for (ep_idx, phy) in tqdm(enumerate(self._storage["physics"])):
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
            self._storage["reward"][ep_idx] = reward
        self._max_episodes = len(self._storage["physics"])
        self._full = True

    # def __iter__(self) -> ReplayBufferIterable:
    #     ''' Returns the Iterator object '''
    #     return ReplayBufferIterable(self)

    # def __iter__(self) -> tp.Iterator[EpisodeBatch[np.ndarray]]:
    #     while True:
    #         yield self.sample()
