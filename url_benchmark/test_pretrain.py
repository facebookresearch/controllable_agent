# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import sys
from pathlib import Path
import subprocess
import pytest


def _run(tmp_path: Path, **params: tp.Any) -> None:
    folder = Path(__file__).parents[1] / "url_benchmark"
    assert folder.exists()
    if sys.platform == "darwin":
        pytest.skip(reason="Does not run on Mac")
    string = " ".join(f"{x}={y}" for (x, y) in params.items())
    command = (
        f"python -m url_benchmark.pretrain device=cpu hydra.run.dir={tmp_path} final_tests=0 "
        + string
    )
    print(f"Running: {command}")
    subprocess.check_call(command.split())


@pytest.mark.parametrize(
    "agent", ["aps", "diayn", "rnd", "proto"]
)  # test most important ones
def test_pretrain_from_commandline(agent: str, tmp_path: Path) -> None:
    _run(
        tmp_path,
        agent=agent,
        num_train_frames=1011,
        num_eval_episodes=1,
        num_seed_frames=1010,
        replay_buffer_episodes=2,
    )


def test_pretrain_from_commandline_fb_with_goal(tmp_path: Path) -> None:
    _run(
        tmp_path,
        agent="fb_ddpg",
        num_train_frames=1,
        num_eval_episodes=1,
        replay_buffer_episodes=2,
        goal_space="simplified_walker",
        use_hiplog=True,
    )
    assert (tmp_path / "hip.log").exists()
