# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from . import cheetah
from . import walker
from . import hopper
from . import quadruped
from . import jaco
from . import point_mass_maze


def make(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False):

    if domain == 'cheetah':
        return cheetah.make(task,
                            task_kwargs=task_kwargs,
                            environment_kwargs=environment_kwargs,
                            visualize_reward=visualize_reward)
    elif domain == 'walker':
        return walker.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'hopper':
        return hopper.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'quadruped':
        return quadruped.make(task,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)
    elif domain == 'point_mass_maze':
        return point_mass_maze.make(task,
                                    task_kwargs=task_kwargs,
                                    environment_kwargs=environment_kwargs,
                                    visualize_reward=visualize_reward)

    else:
        raise ValueError(f'{task} not found')

    assert None


def make_jaco(task, obs_type, seed) -> tp.Any:
    return jaco.make(task, obs_type, seed)
