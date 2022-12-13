# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import io
import random
import traceback
import typing as tp
from pathlib import Path
from collections import defaultdict
import dataclasses

import numpy as np
import torch
from torch.utils.data import IterableDataset
from dm_env import specs, TimeStep


EpisodeTuple = tp.Tuple[np.ndarray, ...]
Episode = tp.Dict[str, np.ndarray]
T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """For later use
    A container for batchable replayed episodes
    """
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    goal: tp.Optional[T] = None
    next_goal: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None
    future_goal: tp.Optional[T] = None

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.reward, (np.ndarray, torch.Tensor))
        assert isinstance(self.discount, (np.ndarray, torch.Tensor))
        assert isinstance(self.meta, dict)

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance on the appropriate device"""
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {x: torch.as_tensor(y, device=device) for x, y in data.items()}  # type: ignore
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                out[field.name] = torch.as_tensor(data, device=device)  # type: ignore
            elif data is None:
                out[field.name] = data
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
        return EpisodeBatch(**out)

    @classmethod
    def collate_fn(cls, batches: tp.List["EpisodeBatch[T]"]) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance from several by stacking in a new first dimension
        for all attributes
        """
        out: tp.Dict[str, tp.Any] = {}
        if isinstance(batches[0].obs, np.ndarray):  # move everything to pytorch if first one is numpy
            batches = [b.to("cpu") for b in batches]  # type: ignore
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in batches]
            # skip fields with None data
            if data[0] is None:
                if any(x is not None for x in data):
                    raise RuntimeError("Found a non-None value mixed with Nones")
                out[field.name] = None
                continue
            # reward and discount can be float which should be converted to
            # tensors for stacking
            if field.name == "meta":
                meta = {k: torch.stack([d[k] for d in data]) for k in data[0]}
                out[field.name] = meta
            elif isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
                # out[field.name] = [x for y in data for x in y]
        return EpisodeBatch(**out)

    def unpack(self) -> tp.Tuple[T, T, T, T, T]:
        """Unpacks the structure into the legacy unnamed tuple.
        Try to avoid it if possible, this is more likely to be wrong than using names
        """
        return (self.obs, self.action, self.reward, self.discount, self.next_obs)
        # return (self.obs, self.action, self.reward, self.discount, self.next_obs, *self.meta)

    def with_no_reward(self: B) -> B:
        reward = self.reward
        reward = torch.zeros_like(reward) if isinstance(reward, torch.Tensor) else 0 * reward
        return dataclasses.replace(self, reward=reward)


def episode_len(episode: Episode) -> int:
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode: Episode, fn: Path) -> None:
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn: Path) -> Episode:
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


Specs = tp.Sequence[specs.Array]


class ReplayBufferStorage:
    def __init__(self, data_specs: Specs, replay_dir: tp.Union[str, Path]) -> None:
        self._data_specs = tuple(data_specs)
        self._meta_specs: tp.Tuple[tp.Any, ...] = tuple()  # deactivated
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(exist_ok=True)
        # probably bad annotation, let's update when it starts failing
        self._current_episode: tp.Dict[str, tp.List[np.ndarray]] = defaultdict(list)
        self._preload()
        raise Exception("This code is dead due to missing handling of meta data")

    def __len__(self) -> int:
        return self._num_transitions

    def add(self, time_step: TimeStep, meta: tp.Mapping[str, np.ndarray]) -> None:
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = {}
            for spec in self._data_specs:
                values = self._current_episode[spec.name]
                episode[spec.name] = np.array(values, spec.dtype)
            for spec in self._meta_specs:
                values = self._current_episode[spec.name]
                episode[spec.name] = np.array(values, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self) -> None:
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode: Episode) -> None:
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, storage: ReplayBufferStorage, max_size: int, num_workers: int, nstep: int, discount: float,
                 fetch_every: int, save_snapshot: bool, future: bool) -> None:
        super().__init__()
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns: tp.List[Path] = []
        self._episodes: tp.Dict[Path, Episode] = {}
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._future = future

    def _sample_episode(self) -> Episode:
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn: Path) -> bool:
        try:
            episode = load_episode(eps_fn)
        except Exception:  # pylint: disable=broad-except
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)  # type: ignore
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)  # type: ignore
        return True

    def _try_fetch(self) -> None:
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = int(torch.utils.data.get_worker_info().id)
        except Exception:  # pylint: disable=broad-except
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes:
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self) -> EpisodeBatch[np.ndarray]:
        try:
            self._try_fetch()
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        meta = {spec.name: episode[spec.name][idx - 1] for spec in self._storage._meta_specs}
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        goal: tp.Optional[np.ndarray] = None
        future_obs: tp.Optional[np.ndarray] = None
        future_goal: tp.Optional[np.ndarray] = None
        if 'goal' in episode.keys():
            goal = episode['goal'][idx - 1]
            if self._future:
                future_idx = idx + np.random.randint(0, episode_len(episode) - idx + 1)
                future_goal = episode['goal'][future_idx - 1]
            # return (obs, goal, action, reward, discount, next_obs, *meta)  # type: ignore
        elif self._future:
            future_idx = idx + np.random.randint(0, episode_len(episode) - idx + 1)
            future_obs = episode['observation'][future_idx - 1]
        # TODO remove type ignore when working
        return EpisodeBatch(obs=obs, action=action, reward=reward, discount=discount,
                            next_obs=next_obs, goal=goal, future_obs=future_obs,
                            future_goal=future_goal, meta=meta)

    def __iter__(self) -> tp.Iterator[EpisodeBatch[np.ndarray]]:
        while True:
            yield self._sample()


def _worker_init_fn(worker_id: int) -> None:
    seed = np.random.get_state()[1][0] + worker_id  # type: ignore
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage: ReplayBufferStorage, max_size: int, batch_size: int, num_workers: int,
                       save_snapshot: bool, future: bool, nstep: int, discount: float) -> tp.Iterable[EpisodeBatch[torch.Tensor]]:
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(storage,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            future=future)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         collate_fn=EpisodeBatch.collate_fn,
                                         worker_init_fn=_worker_init_fn)
    return loader
