# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import tempfile
import typing as tp
from pathlib import Path
import pytest
import hydra
import numpy as np
from url_benchmark import hiplogs
from url_benchmark import utils


def test_until_repr() -> None:
    until = utils.Until(3, 1)
    assert str(until) == "Until(action_repeat=1, until=3)"


def test_parse_logs() -> None:
    path = (
        Path(__file__).parents[1]
        / "controllable_agent"
        / "data"
        / "mockpretrain"
        / "hip.log"
    )
    hlog = hiplogs.HipLog(path)
    logs = hlog.to_hiplot_experiment().datapoints
    assert len(logs) == 13
    vals = logs[-1].values
    assert vals["workdir"] == "054238_fb_ddpg", "Xp id not exported"
    bad_type = {x: y for x, y in vals.items() if not isinstance(y, (int, float, str))}
    assert not bad_type, "Found unsupported type(s)"


def test_load() -> None:
    xp = hiplogs.load(str(Path(__file__).parents[1] / "controllable_agent"), step=2)
    assert len(xp.datapoints) == 6


def test_hiplog(tmp_path: Path) -> None:
    hiplog = hiplogs.HipLog(tmp_path / "log.txt")
    hiplog(hello="world")
    hiplog.write()
    hiplog(hello="monde")
    hiplog(number=12).write()
    hiplog(something=np.int32(12)).write()
    data = hiplog.read()
    for d in data:
        for key in list(d):
            if key.startswith("#"):
                d.pop(key)
    expected = [
        dict(hello="world"),
        dict(hello="monde", number=12),
        dict(hello="monde", number=12, something=12),
    ]
    assert data == expected
    # reloaded
    assert not hiplog._reloaded
    hiplog = hiplogs.HipLog(tmp_path / "log.txt")
    assert hiplog._reloaded == 1


def test_hiplog_stats(tmp_path: Path) -> None:
    hiplog = hiplogs.HipLog(tmp_path / "log.txt")
    for vals in ([3, 5], [7, 8, 9]):
        for val in vals:
            hiplog.with_stats("mean")(val=val)
        hiplog.write()
    data = hiplog.read()
    for d in data:
        for key in list(d):
            if key.startswith("#"):
                d.pop(key)
    expected = [{"val#mean": 4}, {"val#mean": 8}]
    assert data == expected


def test_repository_information() -> None:
    out = hiplogs.repository_information()
    assert len(out) == 3


def test_hiplogs_from_hydra_config(tmp_path: Path) -> None:
    if sys.platform == "darwin":
        pytest.skip(reason="Does not run on Mac")
    train_cmd = [
        sys.executable,
        "-m",
        "url_benchmark.test_hiplogs",
        f"hydra.run.dir={tmp_path}",
    ]
    subprocess.check_call(train_cmd)


@hydra.main(config_name="base_config", config_path=".", version_base="1.1")
def main(args: tp.Any) -> None:
    args.agent.obs_type = "blublu"
    args.agent.obs_shape = (2, 2)
    args.agent.action_shape = (2, 2)
    args.agent.num_expl_steps = 12
    with tempfile.TemporaryDirectory() as tmp:
        log = hiplogs.HipLog(Path(tmp) / "hiplog.test.log").flattened(args)
        assert "agent/obs_type" in log.content


if __name__ == "__main__":
    # needed to load the config:
    from url_benchmark import pretrain  # pylint: disable=unused-import,import-outside-toplevel
    main()
