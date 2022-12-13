# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from url_benchmark.dmc import TimeStep
from url_benchmark.in_memory_replay_buffer import ReplayBuffer

from typing import List

import numpy as np
from dm_env import StepType
import pytest

fixed_episode_lengths = [10, 10, 10, 10, 10]
variable_episode_lengths = [2, 3, 5, 6, 7]
@pytest.mark.parametrize('test_data', [(10, fixed_episode_lengths, None, False, 10),
                                       (5, fixed_episode_lengths, None, True, 10),
                                       (10, variable_episode_lengths, 8, False, 5),
                                       (5, variable_episode_lengths, 8, True, 5)])
def test_avg_episode_length_fixed_length_not_full(test_data) -> None:
    max_episodes, episode_lengths, max_episode_length, is_full, avg_episode_length = test_data
    replay_storage = ReplayBuffer(
        max_episodes=max_episodes, discount=1, future=1, max_episode_length=max_episode_length)
    meta = {'z': np.ones((3, 3))}
    for episode_length in episode_lengths:
        for time_step in _create_dummy_episode(episode_length):
            replay_storage.add(time_step, meta=meta)
    assert replay_storage._full == is_full
    assert replay_storage.avg_episode_length == avg_episode_length

@pytest.mark.parametrize('test_data', [(10, 5, 7), (10, 10, 7)])
def test_backward_compatibility(test_data) -> None:
    max_episodes, episodes_count, episode_length  = test_data
    is_full = max_episodes == episodes_count
    replay_storage = ReplayBuffer(max_episodes=max_episodes, discount=1, future=1, max_episode_length=episode_length + 1)    
    meta = {'z': np.ones((3, 3))}
    for _ in range(episodes_count):
        for time_step in _create_dummy_episode(episode_length):
            replay_storage.add(time_step, meta=meta)
    # remove attributes recently added
    del replay_storage._episodes_length
    del replay_storage._episodes_selection_probability
    del replay_storage._is_fixed_episode_length
    del replay_storage._max_episode_length
    
    loaded_replay_storage = pickle.loads(pickle.dumps(replay_storage))
    assert loaded_replay_storage._idx == episodes_count%max_episodes
    assert loaded_replay_storage._full == is_full
    assert (loaded_replay_storage._episodes_length[:episodes_count]==episode_length).all()
    assert (loaded_replay_storage._episodes_length[episodes_count:]==0).all()
    assert loaded_replay_storage._max_episode_length is None
    

def _create_dummy_episode(episode_length: int) -> List[TimeStep]:
    time_steps = []
    for i in range(episode_length+1):
        step_type = StepType.MID
        if i == 0:
            step_type = StepType.FIRST
        elif i == episode_length:
            step_type = StepType.LAST
        time_step = TimeStep(step_type=step_type, observation=np.zeros(
            (3, 3)), reward=1, discount=1)
        time_steps.append(time_step)
    return time_steps
