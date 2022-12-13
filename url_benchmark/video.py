# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
import cv2
import imageio
import numpy as np
import wandb


class VideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20,
                 camera_id: int = 0,
                 use_wandb: bool = False) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        if self.enabled:
            if hasattr(env, 'physics'):
                if env.physics is not None:
                    frame = env.physics.render(height=self.render_size,
                                               width=self.render_size,
                                               camera_id=self.camera_id)
                else:
                    frame = env.base_env.render()
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name: str) -> None:
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore


class TrainVideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20,
                 camera_id: int = 0,
                 use_wandb: bool = False) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'train_video'
            self.save_dir.mkdir(exist_ok=True)

        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, obs, enabled=True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs) -> None:
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name) -> None:
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore
