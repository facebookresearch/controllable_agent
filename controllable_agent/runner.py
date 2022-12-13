# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import uuid
import shutil
import logging
import datetime
import importlib
import traceback
import contextlib
import typing as tp
from pathlib import Path
import numpy as np
import submitit
import omegaconf
import hydra
from .executor import (  # pylint: disable=unused-import
    DelayedExecutor as DelayedExecutor,
)


PathLike = tp.Union[str, Path]
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(path: tp.Union[str, Path]) -> tp.Iterator[None]:
    cwd = Path().cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class HydraEntryPoint:
    """Creates a callable from a Hydra main
    Config and python files are expected to be in the same folder

    Parameter
    ---------
    script_path: str/Path
        Path to the python script containing main
    """

    # callable to be typed when using an actual package
    def __init__(self, script_path: PathLike) -> None:
        self._script_path = Path(script_path).absolute()
        assert self._script_path.suffix == ".py"
        assert self._script_path.is_file(), f"{self._script_path} is not a file"
        assert self._script_path.with_name("base_config.yaml").is_file()
        self._folder: tp.Optional[Path] = None  # defined later

    @property
    def folder(self) -> Path:
        if self._folder is None:
            raise RuntimeError(
                "Folder is not defined if call method has not be called yet"
            )
        return self._folder

    def validated(self, **kwargs: tp.Any) -> "HydraEntryPoint":
        self._folder = (
            None  # reset folder if validated to avoid reusing a previous test folder
        )
        self.config(**kwargs)
        return self

    def _relative_path(self) -> Path:
        return Path(os.path.relpath(self._script_path, Path(__file__).parent))

    def config(self, **kwargs: tp.Any) -> omegaconf.DictConfig:
        self._get_module()  # needs to be loaded to make sure configs are available
        name = self._script_path.stem
        rel_path = self._relative_path().with_name("base_config.yaml")
        overrides = [f"{x}={y}" for x, y in kwargs.items()]
        with hydra.initialize(
            config_path=str(rel_path.parent), job_name=name, version_base="1.1"
        ):
            cfg_ = hydra.compose(config_name="base_config", overrides=overrides)
        return cfg_

    def _get_module(self) -> tp.Any:
        benchpath = str(self._script_path.parents[1])
        if benchpath not in sys.path:
            sys.path.insert(0, benchpath)
            # add url_benchmark, for legacy buffers
            sys.path.append(str(self._script_path.parent))
        already_imported = any("url_benchmark" in x for x in sys.modules)
        module = importlib.import_module("url_benchmark." + self._script_path.stem)
        module = importlib.reload(module)  # reload to override hydra configstore
        assert module is not None
        if module.__file__ is None or not module.__file__.startswith(benchpath):
            if already_imported:
                logger.warning(
                    "url_benchmark had already been imported, using {module.__file__}"
                )
            else:
                raise RuntimeError(
                    f"Imported {module.__file__} while expecting to be in {benchpath}"
                )
        return module

    def main(self, **kwargs: tp.Any) -> tp.Any:
        return self._get_module().main(self.config(**kwargs))

    def workspace(self, **kwargs: tp.Any) -> tp.Any:
        return self._get_module().Workspace(self.config(**kwargs))

    def __repr__(self) -> str:
        rel_path = str(self._relative_path())
        return f"{self.__class__.__name__}({rel_path!r})"

    def get_hiplog(self) -> tp.Any:
        if self._folder is None:
            raise RuntimeError("No workspace avaible")
        import hiplogs  # type: ignore

        loggers = list(hiplogs.HipLog.find_in_folder(self._folder))
        assert len(loggers) == 1
        return loggers[0]

    def __call__(
        self, _working_directory_: tp.Optional[PathLike] = None, **kwargs: tp.Any
    ) -> float:
        config = self.config(**kwargs)
        try:
            slurm_folder: tp.Optional[Path] = submitit.JobEnvironment().paths.folder
        except RuntimeError:
            slurm_folder = None
        if self._folder is None and _working_directory_ is not None:
            self._folder = Path(_working_directory_)  # override working directory
            self._folder.mkdir(exist_ok=True, parents=True)
            logger.warning(
                f"Bypassing folder affectation and using provided: {self._folder}"
            )
            if slurm_folder is not None:
                # try and link to latest slurm dir anyway
                symlink = self._folder / "slurm"
                if symlink.exists():
                    symlink.unlink()
                symlink.symlink_to(slurm_folder)
        if self._folder is None:
            if slurm_folder is not None:
                self._folder = slurm_folder
            else:
                name = f"{datetime.date.today().isoformat()}_{config.experiment}_{uuid.uuid4().hex[:6]}"
                self._folder = Path("exp_local") / name
        self._folder.mkdir(exist_ok=True, parents=True)
        omegaconf.OmegaConf.save(config=config, f=str(self.folder / "config.yaml"))
        with working_directory(self.folder):
            workspace = self._get_module().Workspace(config)
        try:
            workspace.train()
        except Exception as e:
            if not workspace.eval_rewards_history:
                raise e  # it did not even run :s
            logger.warning(f"Something went wrong:\n{traceback.format_exc()}")
        reward = -float("inf")
        if workspace.eval_rewards_history:
            reward = np.mean(workspace.eval_rewards_history[-12:])
        return -float(reward)  # minimization for nevergrad

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class CopiedBenchmark(HydraEntryPoint):
    def __init__(self, folder: PathLike, name: str) -> None:
        self.code = Path(folder) / "code"
        self.code.parent.mkdir(parents=True, exist_ok=True)
        if self.code.exists():
            logger.warning(
                f"Folder {folder} already exists, it will **not** be updated"
            )
        else:
            shutil.copytree(
                Path(__file__).parents[1] / "url_benchmark",
                self.code / "url_benchmark",
                ignore=shutil.ignore_patterns("exp_*"),
            )
        super().__init__(self.code / "url_benchmark" / f"{name}.py")


def on_exception_enter_postmortem(f):
    """Decorator for triggering pdb in case of exception"""
    import pdb
    import sys
    from functools import wraps
    import traceback

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            pdb.post_mortem(sys.exc_info()[2])
            raise

    return wrapper
