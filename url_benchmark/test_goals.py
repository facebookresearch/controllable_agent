# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import collections
import numpy as np
import pytest

from url_benchmark import goals


def test_basics() -> None:
    assert "simplified_walker" in goals.goal_spaces.funcs["walker"]
    assert len(goals.goals.funcs["simplified_walker"]["walker_stand"]()) == 3


@pytest.mark.parametrize("domain,space", [(d, s) for d in goals.goal_spaces.funcs for s in goals.goal_spaces.funcs[d]])
def test_goal_space_extraction(domain: str, space: str) -> None:
    env = goals._make_env(domain)
    out = goals.goal_spaces.funcs[domain][space](env)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    for name, func in goals.goals.funcs.get(space, {}).items():
        goal = func()
        assert goal.shape == out.shape, f"Wrong shape for goal {name}"
        assert goal.dtype == np.float32


@pytest.mark.parametrize("case", (range(goals.QuadrupedReward.NUM_CASES)))
def test_quad_rewards(case: int) -> None:
    reward = goals.QuadrupedReward()
    reward._case = case
    out = reward.from_physics(reward._env.physics.get_state())
    assert 0 <= out <= 1


def test_quad_pos_rewards() -> None:
    reward = goals.QuadrupedPosReward()
    env = goals._make_env("quadruped")
    env.reset()
    out = reward.from_physics(env.physics.get_state())
    out2 = reward.from_env(env)
    assert 0 <= out <= 1
    assert out == out2, "Should be deterministic"
    assert reward.get_goal("quad_pos_speed").dtype == np.float32


def test_walker_equation() -> None:
    reward = goals.WalkerEquation("1 / (1 + abs(x - 2))")
    env = goals._make_env("walker")
    env.reset()
    out = reward.from_physics(env.physics.get_state())
    out2 = reward.from_env(env)
    assert 0 <= out <= 1
    assert out == out2, "Should be deterministic"


def test_walker_bad_equation() -> None:
    with pytest.raises(ValueError):
        goals.WalkerEquation("1 / (1 + os(x - 2))")


def test_walker_random_equation() -> None:
    env = goals._make_env("walker")
    reward = goals.WalkerRandomReward()
    out = reward.from_env(env)
    assert 0 <= out <= 1


def test_dmc_rewards() -> None:
    env = goals._make_env("quadruped")
    reward = env.task.get_reward(env.physics)
    rewarders = {name: goals.get_reward_function(f"quadruped_{name}") for name in ["walk", "stand"]}
    rewards = {name: r.from_env(env) for name, r in rewarders.items()}
    assert rewards["stand"] == reward
    assert rewards["walk"] != reward
    assert rewarders["stand"].from_physics(env.physics.get_state()) == reward


def test_walker_qpos() -> None:
    env = goals._make_env("walker")
    env.reset()
    env.step(np.random.uniform(-1, 1, size=6))
    out = goals.goal_spaces.funcs["walker"]["walker_pos_speed"](env)
    qpos = env.physics.data.qpos
    assert pytest.approx(qpos[1]) == out[-1], qpos


@pytest.mark.parametrize("name,expected", [("walker_pos_speed", 4)])
def test_goal_space_dim(name: str, expected: int) -> None:
    out = goals.get_goal_space_dim(name)
    assert out == expected


def test_uniquely_named_goal_space() -> None:
    space_counts = collections.Counter(space for spaces in goals.goal_spaces.funcs.values() for space in spaces)
    duplicated = {x for x, y in space_counts.items() if y > 1}
    if duplicated:
        raise RuntimeError(f"Duplicated goal space names: {duplicated}\n(goal space names need to be unique)")


@pytest.mark.parametrize(
    "string,expected", [
        ("(x + y) * z", {"x", "y", "z"}),
        ("import x;os.system(stuff) # hello", {"import", "x", "os", "system", "stuff"}),
    ])
def test_extract_variables(string: str, expected: tp.Set[str]) -> None:
    out = goals.extract_names(string)
    assert out == expected
