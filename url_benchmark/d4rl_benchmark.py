# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import dataclasses
import typing as tp

import numpy as np

from url_benchmark.agent.ddpg import MetaDict
from url_benchmark.dmc import EnvWrapper, ExtendedTimeStep, TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer

from dm_env import specs, StepType
from dm_env.auto_reset_environment import AutoResetEnvironment

@dataclasses.dataclass
class D4RLConfig:
    minimum_episode_length: tp.Optional[int] = None
    ignore_terminals: bool = False

class EmptyPhysics():
    def __init__(self) :
        self.empty_physics = np.zeros((1,1))
    def get_state(self) -> np.ndarray:
        return self.empty_physics

@dataclasses.dataclass
class ExtendedTimeStepD4RL(ExtendedTimeStep):
    reward: tp.Any
    discount: tp.Any

class D4RLWrapper(AutoResetEnvironment):

    def __init__(self, env) -> None:
        self.physics = EmptyPhysics()
        super().__init__()
        self._env = env

    def observation_spec(self) -> tp.Any:
        return specs.BoundedArray(shape=self._env.observation_space.shape,
                                                          dtype=self._env.observation_space.dtype,
                                                          minimum=self._env.observation_space.low,
                                                          maximum=self._env.observation_space.high,
                                                          name='observation')

    def action_spec(self) -> specs.Array:
        return specs.BoundedArray(shape=self._env.action_space.shape,
                                                          dtype=self._env.action_space.dtype,
                                                          minimum=self._env.action_space.low,
                                                          maximum=self._env.action_space.high,
                                                          name='action')
    
    def get_normalized_score(self, reward: float) -> float:
        return self._env.get_normalized_score(reward)

    def _step(self, action) -> TimeStep:
        obs, reward, done, _ = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        return ExtendedTimeStepD4RL(step_type=step_type,observation=obs,reward=reward,discount=1.0,action=action)

    def _reset(self) -> TimeStep:
        obs = self._env.reset()
        return ExtendedTimeStepD4RL(step_type=StepType.FIRST, observation=obs, reward= None, discount= None, action=self._env.action_space.sample())

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, D4RLWrapper):
            return env.base_env
        return env

    def get_dataset(self) -> tp.Any:
        return self._env.get_dataset()

class D4RLReplayBufferBuilder:

    def filter_dataset_by_episode_length(self, dataset: tp.Any, minimum_episode_length: tp.Optional[int]):
        if minimum_episode_length is None or minimum_episode_length <= 1:
            return dataset
        end_indices = (dataset["terminals"] + dataset["timeouts"]).nonzero()[0]
        episode_lengths = np.diff(np.concatenate(([-1], end_indices)))

        episode_lengths_expanded = episode_lengths.repeat(episode_lengths)
        diff_len = dataset['observations'].shape[0] - len(episode_lengths_expanded)
        assert diff_len >= 0 # there is no guarantee that last step in the data step is last step in an episode
        episode_lengths_expanded = np.concatenate((episode_lengths_expanded, np.zeros(diff_len, dtype=int))) # ignore last steps if they do not belong to an episode
        filter_indices = episode_lengths_expanded >= minimum_episode_length
        dataset_size = len(dataset["observations"])
        for key, values in dataset.items():
            if isinstance(values, np.ndarray) and len(values) == dataset_size:
                dataset[key] = values[filter_indices] 

        return dataset
    
    def prepare_replay_buffer_d4rl(self, env: EnvWrapper, meta: MetaDict, cfg: tp.Any) -> ReplayBuffer:
        dataset = env.base_env.get_dataset()
        dataset = self.filter_dataset_by_episode_length(dataset, cfg.d4rl_config.minimum_episode_length)
        # please note we can use d4rl.qlearning_dataset instead, but termination conditions are not calculated as expected only consider (terminals)
        # last next_obs, I used first obs (I see they neglect it at qlearning_dataset, but this will result that last episode will not be terminiated, however we can fake it)
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        is_ignore_terminals = cfg.d4rl_config and (cfg.d4rl_config.ignore_terminals)
        terminals = np.zeros_like(dataset['terminals']) if is_ignore_terminals else dataset['terminals']
        timeouts = dataset['timeouts']
        end_indices = (terminals + timeouts).nonzero()[0]
        episode_lengths = np.diff(np.concatenate(([-1], end_indices)))

        max_episode_length = episode_lengths.max()
        if not cfg.d4rl_config or cfg.d4rl_config.minimum_episode_length is None:
            assert (episode_lengths==1).sum()==0
        else:
            assert (episode_lengths<cfg.d4rl_config.minimum_episode_length).sum()==0 
        replay_storage = ReplayBuffer(max_episodes=len(end_indices), discount=cfg.discount, future=cfg.future, max_episode_length=max_episode_length)
        first = True
        dataset_len = dataset['rewards'].shape[0]
        for idx in range(dataset_len):
            if first:
                time_step = ExtendedTimeStep(
                    step_type = StepType.FIRST, observation=observations[idx], reward=0, discount=1, action=actions[0])
                first = False
            else:
                time_step = ExtendedTimeStep(
                    step_type = StepType.MID, observation=observations[idx], reward=rewards[idx-1], discount=1, action=actions[idx-1])
            
            if terminals[idx] or timeouts[idx]:
                first = True
                final_discount = 1
                if terminals[idx]:
                    final_discount = 0
                time_step.step_type = StepType.LAST
                time_step.discount = final_discount

            replay_storage.add(time_step, meta)
        assert (episode_lengths-1 == replay_storage._episodes_length).all()
        if episode_lengths.min()!=episode_lengths.max():
            assert not replay_storage._is_fixed_episode_length 
        return replay_storage        
