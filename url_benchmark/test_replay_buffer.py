# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from url_benchmark import replay_buffer as rb


def test_batch() -> None:
    shapes = dict(obs=(4, 12), action=(5, 11), next_obs=(6, 10))
    meta = dict(a=np.random.rand(16), b=np.random.rand(17))
    batch = rb.EpisodeBatch(
        reward=np.array([1.0]),
        discount=np.array([0.5]),
        meta=meta,
        **{x: np.random.rand(*y) for x, y in shapes.items()}
    )
    batches = rb.EpisodeBatch.collate_fn([batch, batch])
    assert batches.obs.shape == (2, 4, 12)
    assert isinstance(batches.meta, dict)
    assert len(batches.meta) == 2
    assert batches.meta["a"].shape == (2, 16)
    # check that moving to Tensor does not change anything
    cpu = batch.to("cpu")
    assert cpu.reward.shape == (1,)
    batches = rb.EpisodeBatch.collate_fn([cpu, cpu])
    assert batches.reward.shape == (2, 1)
    no_reward = batches.with_no_reward()
    assert not no_reward.reward.abs().sum(), "reward should be masked"
    assert batches.reward.abs().sum(), "reward should not be masked"
    assert no_reward.obs is batches.obs, "Observations have been copied, which is time consuming"
