# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .ddpg import DDPGAgent as DDPGAgent
from .ddpg import DDPGAgentConfig as DDPGAgentConfig
from .fb_ddpg import FBDDPGAgent as FBDDPGAgent
from .discrete_fb import DiscreteFBAgent as DiscreteFBAgent
from .aps import APSAgent as APSAgent
from .ddpg import MetaDict as MetaDict
# register agents for hydra
from .sf import SFAgent
from .goal_td3 import GoalTD3Agent
from .discrete_sf import DiscreteSFAgent
from .rnd import RNDAgent
from .diayn import DIAYNAgent
from .aps import APSAgent
from .proto import ProtoAgent
from .icm_apt import ICMAPTAgent
from .sf_svd import SFSVDAgent
from .new_aps import NEWAPSAgent
from .goal_sm import GoalSMAgent
from .max_ent import MaxEntAgent
from .exploration import ExplorationAgent
from .uvf import UVFAgent
from .discrete_fb import DiscreteFBAgent
