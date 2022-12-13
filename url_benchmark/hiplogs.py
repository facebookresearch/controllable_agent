# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import collections
from collections import abc
from concurrent import futures
import time
import uuid
import json
import typing as tp
import logging
from datetime import datetime
import subprocess
from pathlib import Path
try:
    from typing import Protocol
except ImportError:
    # backward compatible
    from typing_extensions import Protocol  # type: ignore

import numpy as np
import pandas as pd


# pylint: disable=import-outside-toplevel


START_LINE = "# Hiplot logs"
logger: logging.Logger = logging.getLogger(__name__)


class _StatCall(Protocol):
    def __call__(self, **kwargs: float) -> "HipLog":
        ...


class HipLogfileError(RuntimeError):
    pass


class STYLE:  # pylint: disable=too-few-public-methods
    metrics = "badge badge-pill badge-primary"
    internal = "badge badge-pill badge-secondary"
    optim = "badge badge-pill badge-dark"
    model = "badge badge-pill badge-success"
    other = "badge badge-pill badge-danger"
    # "badge badge-pill badge-warning"


def _set_style(exp: tp.Any) -> None:
    import hiplot as hip

    assert isinstance(exp, hip.Experiment)
    # Don't display `uid` and `from_uid` columns to the user
    cols = set(x for dp in exp.datapoints for x in dp.values.keys())
    internals = ["workdir", "#now", "train/episode", "eval/episode", "#time", "#reloaded", "#job_id"]
    hidden = [x for x in cols if x.startswith(("eval/", "train/"))]
    hidden = [x for x in hidden if not any(y in x for y in ("episode", "loss"))]
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        {
            "hide": ["uid", "from_uid"] + hidden,
        }
    )
    # for the record, some more options:
    exp.display_data(hip.Displays.XY).update(
        {"lines_thickness": 1.4, "lines_opacity": 0.9}
    )
    exp.display_data(hip.Displays.XY).update(
        {"axis_x": "eval/episode", "axis_y": "eval/episode_reward"}
    )
    # colors
    styles = {}
    styles.update(
        {
            name: STYLE.metrics
            for name in cols
            if name.startswith(("eval/", "train/"))
            and not any(y in name for y in ("/episode", "episode_reward"))
        }
    )
    styles.update(
        {name: STYLE.other for name in ("eval/episode_reward", "train/episode_reward")}
    )
    styles.update({name: STYLE.internal for name in internals})
    styles["experiment"] = STYLE.other
    for col in cols:
        for start, style in styles.items():
            if col.startswith(start):
                exp.parameters_definition[col].label_css = style


def create_hiplot_experiment(uri: tp.Union[str, Path]) -> tp.Any:
    import hiplot as hip

    # one xp case
    uri = Path(uri)
    assert uri.suffix == ".csv", f"Path should be a csv, got {uri}"
    assert uri.is_file(), f"Path should be a valid file, but got {uri}"
    df = pd.read_csv(uri)
    prev_uid: tp.Optional[str] = None
    exp = hip.Experiment()
    base = dict(xp=uri.parent.name, date=uri.parents[1].name, mode=uri.stem)
    for k, xp in enumerate(df.itertuples(index=False)):
        data = xp._asdict()
        data.update(base)
        dp = hip.Datapoint(
            uid=f"{uri.parent.name}_{uri.stem}_{k}", from_uid=prev_uid, values=data
        )
        prev_uid = dp.uid
        exp.datapoints.append(dp)
    _set_style(exp)
    return exp


def load(uri: tp.Union[Path, str], step: int = 10) -> tp.Any:
    """Loader for hiplot
    Running:
    python -m hiplot controllable_agent.hiplogs..load --port=XXXX
    will run an hiplot server in which you can past one (or more) log paths
    to plot them
    Note
    ----
    if you install first: "pip install -e ."
    you can simplify to:
    hiplot xxxx.load --port=XXXX
    Then either provide the folder of the experiments in the freeform,
    or their parent directory, so that all subfolders will be parsed for logs.
    """
    import hiplot as hip

    uri = Path(uri)
    if str(uri).startswith("#"):  # deactivate a line
        return hip.Experiment()
    assert uri.is_dir(), f"uri should be a valid directory, got {uri}"
    jobs = []
    with futures.ProcessPoolExecutor() as executor:
        for path in uri.rglob("eval.csv"):
            for hlog in HipLog.find_in_folder(path.parent):
                jobs.append(executor.submit(hlog.to_hiplot_experiment, step))
                # exps.append(create_hiplot_experiment(path))
                # exps.append(create_hiplot_experiment(path.with_name("eval.csv")))
    exps = [j.result() for j in jobs]
    exp = hip.Experiment.merge({str(k): xp for k, xp in enumerate(exps)})
    _set_style(exp)
    return exp


class HipLog:
    """Simple object for logging hiplot compatible content
    Parameters
    ----------
    filepath: str or Path
        path to the logfile. It will be created if it does not exist, otherwise
        data will be appended to it.
    Usage
    -----
    hiplogs are not mutable, adding content is done through
    `with_content` and creates a new instance. This way, you can prefill
    some content, then use the object to add more content and write.
    Example
    -------
    hiplog = hiplogs.HipLog(filepath)
    hiplog = hiplog.with_content(shared_key=12)
    hiplog.write()  # writes only {"shared_key": 12}
    hiplog.with_content(hello="world").write()  # writes shared_key and hello
    hiplog.with_content(something="blublu").write()  # writes shared_key and something
    """

    def __init__(self, filepath: tp.Union[Path, str]) -> None:
        self._filepath = Path(filepath)
        if self._filepath.suffix not in (".txt", ".log"):
            raise ValueError("Filepath must have .txt or .log as extension")
        self._content: tp.Dict[str, tp.Any] = {
            "#start_time": f"{datetime.now():%Y-%m-%d %H:%M}"
        }
        self._floats: tp.Dict[str, tp.List[float]] = collections.defaultdict(list)
        self._stats: tp.Dict[str, tp.Tuple[str, ...]] = {}
        self._reloaded = 0
        try:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            if not self._filepath.exists():
                self._filepath.write_text(START_LINE + " v1\n", encoding="utf8")
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failing to write data to json: %s", e)
        try:
            import submitit
            self._content["#job_id"] = submitit.JobEnvironment().job_id
        except Exception:  # pylint: disable=broad-except
            pass
        data = self.read()
        if data:
            self._reloaded = data[-1].get("#reloaded", -1) + 1  # type: ignore

    @classmethod
    def find_in_folder(
        cls, folder: tp.Union[str, Path], recursive: bool = False
    ) -> tp.Iterator["HipLog"]:
        """Instantiate all hiplog instances from the folder or subfolders

        Parameters
        ----------
        folder: str/Path
            folder to look into
        recursive: bool
            instantiate all hiplog logs recursively

        Yields
        ------
        HipLog
            hiplog instance
        """
        folder = Path(folder)
        for suffix in (".txt", ".log"):
            iterator = (folder.rglob if recursive else folder.glob)("*" + suffix)
            for fp in iterator:
                if fp.suffix in (".log", ".txt"):
                    with fp.open("r", encoding="utf8") as f:
                        is_hiplog = START_LINE in f.readline()
                    if is_hiplog:
                        yield cls(fp)

    def __call__(self, **kwargs: tp.Hashable) -> "HipLog":
        sanitized = {
            x: y if not isinstance(y, np.generic) else y.item()
            for x, y in kwargs.items()
        }
        self._content.update(sanitized)
        return self

    def with_stats(self, *stats: tp.Sequence[str]) -> _StatCall:
        return functools.partial(self._with_stats, tuple(stats))

    def _with_stats(self, _internal_name_stats: tp.Tuple[str, ...], **kwargs: float) -> "HipLog":
        for key, val in kwargs.items():
            self._stats[key] = _internal_name_stats  # overridden by last call
            self._floats[key].append(float(val))
        return self

    def read(self, step: int = 1) -> tp.List[tp.Dict[str, tp.Hashable]]:
        """Returns the data recorded through the logger

        Parameter
        ---------
        step: int
            step for decimating the data if too big

        Returns
        -------
        list of dict
            all the timepoints. Data from past timepoints are used if not
            provided in newer timepoints (eg: initial hyperparameters are
            passed to all timepoints)
        """
        with self._filepath.open("r", encoding="utf8") as f:
            lines = f.readlines()
        if lines and not lines[0].startswith(START_LINE):
            raise HipLogfileError(
                f"Did not recognize first line: {lines[0]!r} instead of {START_LINE!r}"
            )
        data: tp.List[tp.Dict[str, tp.Hashable]] = []
        last = {}
        for k, line in enumerate(lines):
            if not line.startswith("#"):
                line_dict = json.loads(line.strip())
                last.update(line_dict)
                if not k % step:
                    data.append(dict(last))
        return data

    def last_line(self) -> tp.Dict[str, tp.Hashable]:
        data = self.read()
        return {} if not data else data[-1]

    @property
    def content(self) -> tp.Dict[str, tp.Hashable]:
        return dict(self._content)

    def _export_floats(self) -> tp.Dict[str, float]:
        out: tp.Dict[str, float] = {}
        for key, vals in self._floats.items():
            for stat in self._stats[key]:
                out[f"{key}#{stat}"] = getattr(np, stat)(vals)
        return out

    def write(self) -> None:
        # avoid as much as possible any disruption
        self._content["#now"] = f"{datetime.now():%Y-%m-%d %H:%M}"
        self._content["#time"] = time.time()
        self._content["#reloaded"] = self._reloaded
        self._content.update(self._export_floats())
        if not self._filepath.exists():
            return  # initialization failed, can't do anything more
        try:
            string = json.dumps(self._content)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failing to write data to json: %s", e)
            return  # can't be json-ed, stop there
        # if it reaches here, it should be safe to write
        with self._filepath.open("a", encoding="utf8") as f:
            f.write(string + "\n")
        self._content.clear()
        self._floats.clear()
        self._stats.clear()

    def flattened(self, data: tp.Any) -> "HipLog":
        """Flattens a structured configuration and adds it to the content"""
        self(**_flatten(data))
        return self

    def to_hiplot_experiment(self, step: int = 1) -> tp.Any:
        """Returns the Experiment recorded  through the logger

        Parameter
        ---------
        step: int
            step for decimating the data if too big

        Returns
        -------
        Experiment
            Hiplot Experiment instance containing the logger data
        """
        import hiplot as hip
        exp = hip.Experiment()
        prev_uid: tp.Optional[str] = None
        name = uuid.uuid4().hex[:8]
        for k, data in enumerate(self.read(step=step)):
            # update the displayed name to something readable
            if not k:
                xp = data.get("experiment", "#UNKNOWN#")
                job_id = data.get("#job_id", name)
                name = f"{xp} / {job_id}"
            dp = hip.Datapoint(uid=f"{name} / {k}", from_uid=prev_uid, values=data)  # type: ignore
            prev_uid = dp.uid
            exp.datapoints.append(dp)
        _set_style(exp)
        logger.info("Finished loading %s", self._filepath)
        return exp


def _flatten(data: abc.Mapping) -> tp.Dict[str, tp.Hashable]:  # type: ignore
    output: tp.Dict[str, tp.Hashable] = {}
    if isinstance(data, abc.Mapping):
        for x, y in data.items():
            if isinstance(y, abc.Mapping):
                content = _flatten(y)
                output.update({f"{x}/{x2}": y2 for x2, y2 in content.items()})
            elif isinstance(y, abc.Sequence) and not isinstance(y, str):
                if y and isinstance(
                    y[0], (int, float, str)
                ):  # ignoring weird structures
                    output[x] = ",".join(str(z) for z in y)
            elif isinstance(y, abc.Hashable):
                output[x] = y
    return output


def repository_information() -> tp.Dict[str, str]:
    commands = {
        "commit": "git rev-parse --short HEAD",
        "branch": "git rev-parse --abbrev-ref HEAD",
        "closest_main": "git rev-parse --short main",
    }
    here = Path(__file__).parent
    output: tp.Dict[str, str] = {}
    for name, command in commands.items():
        try:
            output[name] = (
                subprocess.check_output(command.split(), shell=False, cwd=here)
                .strip()
                .decode()
            )
        except Exception:  # pylint: disable=broad-except
            pass
    return output
