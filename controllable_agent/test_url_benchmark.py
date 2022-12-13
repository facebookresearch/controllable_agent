# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
import itertools
import subprocess
from pathlib import Path
import controllable_agent
from . import runner


def test_quadruped_goal(tmp_path: Path) -> None:
    conf_path = Path(__file__).parents[1] / "url_benchmark" / "pretrain.py"
    with runner.working_directory(tmp_path):
        ep = runner.HydraEntryPoint(conf_path)
        ep(
            _working_directory_=tmp_path / "bypass",
            agent="fb_ddpg",
            device="cpu",
            num_train_frames=1,
            num_eval_episodes=1,
            replay_buffer_episodes=2,
            goal_space="simplified_quadruped",
            task="quadruped_walk",
            use_hiplog=True,
            final_tests=1,
            **{"agent.feature_dim": 80, "agent.z_dim": 100},
        )
        reward_file = tmp_path / "bypass" / "test_rewards.json"
        text = reward_file.read_text()
        assert "quadruped_run" in text


def test_anytrain(tmp_path: Path) -> None:
    with runner.working_directory(tmp_path):
        ep = runner.CopiedBenchmark(tmp_path / "no_code", "anytrain")
        ep(
            agent="fb_ddpg",
            device="cpu",
            num_train_episodes=1,
            num_eval_episodes=1,
            replay_buffer_episodes=2,
            use_hiplog=True,
            final_tests=0,
        )


def test_grid_anytrain(tmp_path: Path) -> None:
    with runner.working_directory(tmp_path):
        ep = runner.CopiedBenchmark(tmp_path / "no_code", "anytrain")
        ep(
            agent="discrete_fb",
            device="cpu",
            task="grid_simple",
            num_train_episodes=1,
            num_eval_episodes=1,
            replay_buffer_episodes=2,
            use_hiplog=True,
            final_tests=0,
        )


def test_package_init_annotations() -> None:
    # automatically updates the __init__ functions with "-> None:" if missing
    # it fails the first time when adding it, then it should work
    # feel free to deactivate if that helps, it's not that important :p
    failed = []
    pattern = re.compile(r"(def __init__\(self.*\)):")
    root = Path(__file__).parents[1]
    assert (root / "url_benchmark").is_dir()
    for fp in root.rglob("*.py"):
        if "expected" in str(fp) or "test_" in fp.name:
            continue
        text = fp.read_text()
        text2 = pattern.sub(r"\g<1> -> None:", text)
        if text2 != text:
            failed.append(str(fp))
            fp.write_text(text2)
    if failed:
        string = "\n -".join(
            ["Missing -> None at the end of __init__ definition"] + failed
        )
        string += "\nUpdate, or run this test locally for automatic addition"
        raise AssertionError(string)


def test_property_syntax() -> None:
    # automatic linters tend to change @property to @ property for no reason
    root = Path(__file__).parents[1]
    assert (root / "url_benchmark").is_dir()
    errors = []
    for fp in root.rglob("*.py"):
        if fp == Path(__file__):
            continue
        if "@ property" in fp.read_text():
            errors.append(str(fp))
    if errors:
        msg = ["Additional space in @property, linter got crazy:"] + errors
        raise AssertionError("\n  - ".join(msg))


def test_pretrain_checkpoint(tmp_path: Path) -> None:
    conf_path = Path(__file__).parents[1] / "url_benchmark" / "pretrain.py"
    with runner.working_directory(tmp_path):
        ep = runner.HydraEntryPoint(conf_path)
        params = dict(
            agent="fb_ddpg",
            device="cpu",
            num_train_frames=1001,
            num_eval_episodes=1,
            replay_buffer_episodes=2,
            use_hiplog=True,
            checkpoint_every=1000,
            final_tests=0,
        )
        wsp = ep.workspace(**params)
        assert not wsp.global_step
        wsp.train()
        assert wsp.global_step == 1001
        wsp2 = ep.workspace(**params)
        assert wsp2.global_step == 1001


# keep last because it may make a mess with the paths (for copied benchmark)
def test_pretrain_from_runner(tmp_path: Path) -> None:
    conf_path = Path(__file__).parents[1] / "url_benchmark" / "pretrain.py"
    with runner.working_directory(tmp_path):
        ep = runner.HydraEntryPoint(conf_path)
        reward = ep(
            agent="fb_ddpg",
            device="cpu",
            num_train_frames=1011,
            num_eval_episodes=1,
            num_seed_frames=1010,
            replay_buffer_episodes=2,
            use_hiplog=True,
            final_tests=0,
        )
        assert isinstance(reward, float)
        assert -1000 < reward < 0
        from url_benchmark import hiplogs  # pylint: disable=import-outside-toplevel

        hippath = ep.folder / "hip.log"
        assert hippath.exists()
        hiploggers = list(hiplogs.HipLog.find_in_folder(tmp_path, recursive=True))
        assert len(hiploggers) == 1
        hiplog = hiploggers[0]
        out = hiplog.read()
        assert "eval/episode_reward" in out[0]
        confpath = ep.folder / "config.yaml"
        assert confpath.exists()


def test_header() -> None:
    lines = Path(__file__).read_text("utf8").splitlines()
    header = "\n".join(itertools.takewhile(lambda l: l.startswith("#"), lines))
    assert len(header.splitlines()) == 4, f"Identified header:\n{header}"
    root = Path(controllable_agent.__file__).parents[1]
    base = [x for x in root.iterdir() if not x.name.startswith(".")]  # avoid .git
    tocheck = []
    for fp in base:
        if fp.is_file() and fp.suffix == ".py":
            tocheck.append(fp)
        elif fp.is_dir():
            output = subprocess.check_output(
                ["find", str(fp), "-name", "*.py"], shell=False
            )
            tocheck.extend([Path(p) for p in output.decode().splitlines()])
    missing = []
    AUTOADD = True
    for fp in tocheck:
        text = Path(fp).read_text("utf8")
        if not text.startswith(header):
            if AUTOADD and not any(x in text.lower() for x in ("license", "copyright")):
                print(f"Automatically adding header to {fp}")
                Path(fp).write_text(header + "\n\n" + text, "utf8")
            missing.append(str(fp))
    if missing:
        missing_str = "\n - ".join(missing)
        raise AssertionError(
            f"Following files are/were missing standard header (see other files):\n - {missing_str}"
        )
