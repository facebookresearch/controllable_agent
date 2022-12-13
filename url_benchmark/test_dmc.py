# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import pytest
from url_benchmark import dmc
from url_benchmark.in_memory_replay_buffer import ReplayBuffer


@pytest.mark.parametrize("name,expected", [
    ("walker_walk", {'torso_height': 1.3, 'torso_upright': 1.0, 'horizontal_velocity': 0.0}),
    ("quadruped_walk", {'torso_upright': 1.0, 'torso_velocity#0': 0.0, 'torso_velocity#1': 0.0, 'torso_velocity#2': 0.0}),
])
def test_extract_physics(name: str, expected: tp.Dict[str, float]) -> None:
    env = dmc.make(name, obs_type="states", frame_stack=1, action_repeat=1, seed=12)
    phy = dmc.extract_physics(env)
    assert phy == expected
    time_step = env.reset()
    assert time_step.physics.size > 0
    # check that it works in the ReplayBuffer
    rb = ReplayBuffer(12, 0.9, True)
    rb.add(time_step, {})
    assert "physics" in rb._current_episode


def test_goal_wrapper() -> None:
    env = dmc.make("quadruped_walk", obs_type="states", frame_stack=1, action_repeat=1,
                   seed=12, goal_space="simplified_quadruped", append_goal_to_observation=True)
    out = env.reset()
    assert out.observation.shape == env.observation_spec().shape
    env = dmc.make("quadruped_walk", obs_type="states", frame_stack=1, action_repeat=1,
                   seed=12, goal_space="simplified_quadruped", append_goal_to_observation=False)
    out2 = env.reset()
    assert out2.observation.shape[0] < out.observation.shape[0]


def test_physics_aggregator() -> None:
    env = dmc.make("walker_walk", obs_type="states", frame_stack=1, action_repeat=1, seed=12)
    agg = dmc.PhysicsAggregator()
    agg.add(env)
    names = [x[0] for x in agg.dump()]
    assert len(names) == 9
    assert not list(agg.dump())


def test_float_stats() -> None:
    stats = dmc.FloatStats().add(12)
    assert all(getattr(stats, name) == 12 for name in ["mean", "max", "min"])
    stats.add(24)
    assert stats.min == 12
    assert stats.max == 24
    assert stats.mean == 18
    assert stats._count == 2
    stats.add(24)
    assert stats.mean == 20
