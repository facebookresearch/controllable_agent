# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import dataclasses
from types import ModuleType
import numpy as np
import torch
from url_benchmark import replay_buffer as rb
from url_benchmark import agent as agents
from . import fb_ddpg
from . import fb_modules


def get_cfg() -> fb_ddpg.FBDDPGAgentConfig:
    # hopefully this can get simpler soon
    return fb_ddpg.FBDDPGAgentConfig(
        obs_shape=(4,), action_shape=(3,), obs_type="state", device="cpu", num_expl_steps=1, goal_space=None
    )


def test_agent_init() -> None:
    cfg = get_cfg()
    agent = fb_ddpg.FBDDPGAgent(**dataclasses.asdict(cfg))
    b = 12
    shapes = dict(obs=(b, 4), next_obs=(b, 4), action=(b, 4), reward=(b,), discount=(b,))
    iterator = (rb.EpisodeBatch(**{x: np.random.rand(*y).astype(np.float32)
                for x, y in shapes.items()}) for _ in range(100))  # type: ignore
    meta = agent.init_meta()
    with torch.no_grad():
        action = agent.act(next(iterator).obs[0], meta, 0, eval_mode=False)
    assert action.shape == (3,)


def test_agents_config() -> None:
    cfgs = []
    for module in agents.__dict__.values():
        if isinstance(module, ModuleType):
            for obj in module.__dict__.values():
                if inspect.isclass(obj) and issubclass(obj, agents.DDPGAgentConfig):
                    if obj not in cfgs:
                        cfgs.append(obj)
    assert len(cfgs) >= 3
    for cfg in cfgs:
        # check that target and name have been updated to match the algo
        assert cfg.name.replace("_", "") in cfg.__name__.lower()
        assert cfg.name in cfg._target_


def test_multiinputs() -> None:
    m, n = [10, 12]
    x, y = (torch.rand([16, z]) for z in [m, n])
    mip = fb_modules.MultinputNet([m, n], [100, 100, 32])
    out = mip(x, y)
    assert out.shape == (16, 32)
