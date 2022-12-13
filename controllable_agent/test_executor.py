# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import concurrent.futures
from pathlib import Path
import pytest
import submitit
from . import executor as _exec


def func(fail: bool = False) -> str:
    if fail:
        raise ValueError("This is a failure")
    return "success"


def get_executor(tmp_path: Path) -> _exec.DelayedExecutor[str]:
    local_exec = submitit.AutoExecutor(folder=tmp_path, cluster="debug")
    return _exec.DelayedExecutor(
        local_exec, default="ERROR", batch_size=2, max_delay=0.2, max_failure_rate=0.5
    )


def test_delayed_exec_num(tmp_path: Path) -> None:
    executor = get_executor(tmp_path)
    job1 = executor.submit(func)
    assert not job1.done()
    assert job1.job is None, "Job should not be submitted"
    job2 = executor.submit(func)
    assert job2.done()
    assert job1.job is not None, "Job should not be submitted"
    assert job2.job is not None, "Job should not be submitted"
    assert not executor._unsubmitted, "Unsubmitted jobs should be purged"


def test_delayed_exec_delay(tmp_path: Path) -> None:
    executor = get_executor(tmp_path)
    job1 = executor.submit(func)
    time.sleep(0.1)
    assert job1.job is None, "Job should not be submitted"
    time.sleep(0.11)
    job1.done()  # trigger a possible submission
    assert job1.job is not None, "Job should be submitted"
    assert not executor._unsubmitted, "Unsubmitted jobs should be purged"


def test_delayed_exec_error(tmp_path: Path) -> None:
    executor = get_executor(tmp_path)
    jobs = [executor.submit(func, fail=f) for f in [True, True]]
    with pytest.raises(RuntimeError):
        jobs[0].result()


def test_delayed_exec_caught_error(tmp_path: Path) -> None:
    executor = get_executor(tmp_path)
    jobs = [executor.submit(func, fail=f) for f in [False, True]]
    assert jobs[0].result() == "success"
    assert jobs[1].result() == "ERROR"


def _do_nothing() -> int:
    return 12


def test_wait_for_jobs() -> None:
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exc:
        for _ in range(2):
            jobs.append(exc.submit(_do_nothing))
            _exec.wait_for_jobs(jobs, sleep=0.04)
