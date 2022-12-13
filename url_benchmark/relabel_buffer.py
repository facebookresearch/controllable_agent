# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from url_benchmark.goals import DmcReward
import torch

name = "walker_flip"
load_replay_buffer = "/checkpoint/jrapin/ca/buffers/walker_rnd_ddpg_220803.pt"
relabeled_replay_file_path = "/private/home/atouati/controllable_agent/datasets/walker/rnd/walker_flip_rnd_ddpg.pt"
custom_reward = DmcReward(name)

print("loading Replay from %s", load_replay_buffer)
with open(load_replay_buffer, 'rb') as f:
    replay_loader = torch.load(f)
replay_loader.relabel(custom_reward)

with open(relabeled_replay_file_path, 'wb') as f:
    torch.save(replay_loader, f)
