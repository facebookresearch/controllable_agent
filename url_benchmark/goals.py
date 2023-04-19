# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import token
import tokenize
import functools
import typing as tp
from io import BytesIO
from collections import OrderedDict
import numpy as np
from url_benchmark import dmc
from dm_control.utils import rewards

from url_benchmark.custom_dmc_tasks.jaco import TASKS as jaco_tasks_list
from url_benchmark.custom_dmc_tasks.point_mass_maze import TASKS as point_mass_maze_tasks_list
from url_benchmark.in_memory_replay_buffer import ReplayBuffer

jaco_tasks = dict(jaco_tasks_list)
point_mass_maze_tasks = dict(point_mass_maze_tasks_list)

F = tp.TypeVar("F", bound=tp.Callable[..., np.ndarray])


class Register(tp.Generic[F]):

    def __init__(self) -> None:
        self.funcs: tp.Dict[str, tp.Dict[str, F]] = {}

    def __call__(self, name: str) -> tp.Callable[[F], F]:
        return functools.partial(self._register, name=name)

    def _register(self, func: F, name: str) -> F:
        fname = func.__name__
        subdict = self.funcs.setdefault(name, {})
        if fname in subdict:
            raise ValueError(f"Already registered a function {fname} for {name}")
        subdict[fname] = func
        return func


goal_spaces: Register[tp.Callable[[dmc.EnvWrapper], np.ndarray]] = Register()
goals: Register[tp.Callable[[], np.ndarray]] = Register()


# # # # #
# goal spaces, defined on one environment to specify:
# # # # #

# pylint: disable=function-redefined

@goal_spaces("jaco")
def simplified_jaco(env: dmc.EnvWrapper) -> np.ndarray:
    return np.array(env.physics.bind(env.task._hand.tool_center_point).xpos,
                    dtype=np.float32)


@goal_spaces("point_mass_maze")
def simplified_point_mass_maze(env: dmc.EnvWrapper) -> np.ndarray:
    return np.array(env.physics.named.data.geom_xpos['pointmass'][:2],
                    dtype=np.float32)


@goal_spaces("walker")
def simplified_walker(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    return np.array([env.physics.torso_height(),
                     env.physics.torso_upright(),
                     env.physics.horizontal_velocity()],
                    dtype=np.float32)


@goal_spaces("walker")
def walker_pos_speed(env: dmc.EnvWrapper) -> np.ndarray:
    """simplifed walker, with x position as additional variable"""
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    x = env.physics.named.data.xpos['torso', 'x']
    return np.concatenate([simplified_walker(env), [x]], axis=0, dtype=np.float32)  # type: ignore


@goal_spaces("walker")
def walker_pos_speed_z(env: dmc.EnvWrapper) -> np.ndarray:
    """simplifed walker, with x position as additional variable"""
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/walker.py
    # vz = env.physics.named.data.sensordata["torso_subtreelinvel"][-1]
    # om_y = env.physics.named.data.subtree_angmom['torso'][1]
    vz = env.physics.named.data.subtree_linvel['torso', 'z']
    om_y = env.physics.named.data.subtree_angmom['torso', 'y']
    return np.concatenate([walker_pos_speed(env), [vz, om_y]], axis=0, dtype=np.float32)  # type: ignore


@goal_spaces("quadruped")
def simplified_quadruped(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.array([env.physics.torso_upright(),
                     np.linalg.norm(env.physics.torso_velocity())],
                    dtype=np.float32)


@goal_spaces("quadruped")
def quad_pos_speed(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    x = np.array(env.physics.named.data.site_xpos['workspace'])
    states = [[env.physics.torso_upright()], x, env.physics.torso_velocity()]
    return np.concatenate(states, dtype=np.float32)


# @goal_spaces("quadruped")  # this one needs a specific task for the ball to be present
# def quadruped_positions(env: dmc.EnvWrapper) -> np.ndarray:
#     data = env.physics.named.data
#     states = [data.xpos['ball'] - data.site_xpos['target'], data.xpos["torso"] - data.site_xpos['target']]
#     return np.concatenate(states, dtype=np.float32)


# # # # #
# goals, defined on one goal_space to specify:
# # # # #


@goals("simplified_walker")
def walker_stand() -> np.ndarray:
    return np.array([1.2, 1.0, 0], dtype=np.float32)


@goals("simplified_walker")
def walker_walk() -> np.ndarray:
    return np.array([1.2, 1.0, 2], dtype=np.float32)


@goals("simplified_walker")
def walker_run() -> np.ndarray:
    return np.array([1.2, 1.0, 4], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_stand() -> np.ndarray:
    return np.array([1.0, 0], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_walk() -> np.ndarray:
    return np.array([1.0, 0.6], dtype=np.float32)


@goals("simplified_quadruped")
def quadruped_run() -> np.ndarray:
    return np.array([1.0, 6], dtype=np.float32)


@goals("quadruped_positions")
def quadruped_fetch() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_top_left() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_top_left'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_top_right() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_top_right'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_bottom_left() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_bottom_left'],
                    dtype=np.float32)


@goals("simplified_point_mass_maze")
def point_mass_maze_reach_bottom_right() -> np.ndarray:
    return np.array(point_mass_maze_tasks['reach_bottom_right'],
                    dtype=np.float32)


@goals("simplified_jaco")
def jaco_reach_top_left() -> np.ndarray:
    return jaco_tasks['reach_top_left'].astype(np.float32)


@goals("simplified_jaco")
def jaco_reach_top_right() -> np.ndarray:
    return jaco_tasks['reach_top_right'].astype(np.float32)


@goals("simplified_jaco")
def jaco_reach_bottom_left() -> np.ndarray:
    return jaco_tasks['reach_bottom_left'].astype(np.float32)


@goals("simplified_jaco")
def jaco_reach_bottom_right() -> np.ndarray:
    return jaco_tasks['reach_bottom_right'].astype(np.float32)


@goals("walker_pos_speed_z")
def walker_dummy() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)

# # # Custom Reward # # #


def _make_env(domain: str) -> dmc.EnvWrapper:
    task = {"quadruped": "stand", "walker": "walk", "jaco": "reach_top_left", "point_mass_maze": "reach_bottom_right"}[domain]
    return dmc.make(f"{domain}_{task}", obs_type="states", frame_stack=1, action_repeat=1, seed=12)


def get_goal_space_dim(name: str) -> int:
    domain = {space: domain for domain, spaces in goal_spaces.funcs.items() for space in spaces}[name]
    env = _make_env(domain)
    return goal_spaces.funcs[domain][name](env).size


class BaseReward:

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


def get_reward_function(name: str, seed: tp.Optional[int] = None) -> BaseReward:
    if name == "quadruped_mix":
        return QuadrupedReward(seed)
    if name == "walker_random_equation":
        return WalkerRandomReward(seed)
    if name == "quadruped_position":
        return QuadrupedPosReward(seed)
    if name == "maze_multi_goal":
        return MazeMultiGoal(seed)
    if name == "walker_position":
        return WalkerPosReward(seed)
    return DmcReward(name)


def _inv(distance: float) -> float:
    # print("dist", distance)
    return 1 / (1 + abs(distance))


class DmcReward(BaseReward):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        env_name, task_name = name.split("_", maxsplit=1)
        try:
            from dm_control import suite  # import
            from url_benchmark import custom_dmc_tasks as cdmc
        except ImportError as e:
            raise dmc.UnsupportedPlatform("DMC does not run on Mac") from e
        make = suite.load if (env_name, task_name) in suite.ALL_TASKS else cdmc.make
        self._env = make(env_name, task_name)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))

    # def from_env(self, env: dmc.EnvWrapper) -> float:
    #    return self.from_physics(env.physics.get_state())
#
    # def from_physics(self, physics: np.ndarray) -> float:
    #    # pdb.set_trace()
    #    with self._env.physics.reset_context():
    #        self._env.physics.set_state(physics)
    #    return float(self._env.task.get_reward(self._env.physics))


class QuadrupedReward(BaseReward):

    NUM_CASES = 7

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)
        self._env = _make_env("quadruped")
        self.x = self._rng.uniform(-5, 5, size=2)
        self.vx = self._rng.uniform(-3, 3, size=2)
        self.quadrant = self._rng.choice([1, -1], size=2, replace=True)
        self.speed = float(np.linalg.norm(self.vx))
        self._case = self._rng.randint(self.NUM_CASES)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        # x = env.physics.named.data.xpos["torso"][:2]
        x = env.physics.named.data.site_xpos['workspace'][:2]
        vx = env.physics.torso_velocity()[:2]
        up = max(0, float(env.physics.torso_upright()))
        speed = float(np.linalg.norm(vx))
        if not self._case:   # specific speed norm
            return up * _inv(speed - self.speed)
        if self._case == 1:  # specific position
            return up * _inv(float(np.linalg.norm(x - self.x)))
        if self._case == 2:  # specific quadrant
            return up * float(np.all(x * self.quadrant > self.x))
        if self._case == 3:  # specific quadrant and speed norm
            return up * float(np.all(x * self.quadrant > self.x)) * _inv(self.speed - speed)
        if self._case == 4:  # specific speed
            return up * _inv(np.linalg.norm(self.vx - vx) / np.sqrt(2))
        if self._case == 5:  # specific quadrant and sufficient speed
            return up * float(np.all(x * self.quadrant > self.x)) * (speed > self.speed)
        if self._case == 6:  # sufficient speed
            return up * (speed > self.speed)
        else:
            raise ValueError(f"No case #{self._case}")


class QuadrupedPosReward(BaseReward):
    """Deterministic positional reward"""

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)
        self._env = _make_env("quadruped")
        self.x = np.array([2, 2, 0.8])

    def get_goal(self, goal_space: str) -> np.ndarray:
        if goal_space != "quad_pos_speed":
            raise ValueError(f"Goal space {goal_space} not supported with this reward")
        states = [[1.0], self.x, [0] * 3]
        return np.concatenate(states, dtype=np.float32)  # type: ignore

    def from_env(self, env: dmc.EnvWrapper) -> float:
        x = env.physics.named.data.site_xpos['workspace']
        up = float(env.physics.torso_upright())
        up = (up + 1) / 2
        out = 0.5 * up + 0.5 * _inv(float(np.linalg.norm(x - self.x)))  # * _inv(speed)
        return out


class WalkerPosReward(BaseReward):
    """Random positional reward"""

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)
        self._env = _make_env("walker")
        self.x = np.random.randint(-20, 20)

    def get_goal(self, goal_space: str) -> np.ndarray:
        if goal_space != "walker_pos_speed_z":
            raise ValueError(f"Goal space {goal_space} not supported with this reward")
        states = [1, 1, 0, self.x, 0, 0]  # [z, up, vx, x, vz, om_y]
        # states = [self.x]
        return np.array(states, dtype=np.float32)  # type: ignore

    def from_env(self, env: dmc.EnvWrapper) -> float:
        x = env.physics.named.data.xpos['torso', 'x']
        target_size = 1
        d = abs(x - self.x)
        reward = rewards.tolerance(d, bounds=(0, target_size), margin=target_size)
        return reward


class MazeMultiGoal(BaseReward):
    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)

        self.goals = np.array([
            [-0.15, 0.15],  # room 1: top left
            [-0.22, 0.22],  # room 1
            [-0.08, 0.08],  # room 1
            [-0.22, 0.08],  # room 1
            [-0.08, 0.22],  # room 1
            [0.15, 0.15],  # room 2: top right
            [0.22, 0.22],  # room 2
            [0.08, 0.08],  # room 2
            [0.22, 0.08],  # room 2
            [0.08, 0.22],  # room 2
            [-0.15, -0.15],  # room 3: bottom left
            [-0.22, -0.22],  # room 3
            [-0.08, -0.08],  # room 3
            [-0.22, -0.08],  # room 3
            [-0.08, -0.22],  # room 3
            [0.15, -0.15],  # room 4: bottom right
            [0.22, -0.22],  # room 4
            [0.08, -0.08],  # room 4
            [0.22, -0.08],  # room 4
            [0.08, -0.22],  # room 4
        ], dtype=np.float32)
        # save images for debug
        # import imageio
        # self._env = dmc.make("point_mass_maze_multi_goal", obs_type="states", frame_stack=1, action_repeat=1, seed=12)
        # self._env.reset()
        # img = self._env.physics.render(height=256, width=256, camera_id=0)
        # imageio.imsave("maze.png", img)

    def from_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> tp.Tuple[float, float]:
        """returns reward and distance"""
        assert achieved_goal.shape == desired_goal.shape
        target_size = .03
        d: np.ndarray = achieved_goal - desired_goal
        distance = np.linalg.norm(d, axis=-1) if len(d.shape) > 0 else np.linalg.norm(d)
        reward = rewards.tolerance(distance,
                                   bounds=(0, target_size), margin=target_size)
        return reward, distance


class WalkerYogaReward():
    def __init__(self) -> None:
        self._env = _make_env("walker")
        self._goals = get_walkeryoga_goals()
        self.target_obs = {}
        for key, g in self._goals.items():
            self.target_obs[key] = get_obs_from_yoga_goal(self._env, g).astype(np.float32)
            # save images for debug
            # import imageio
            # img = self._env.physics.render(height=256, width=256, camera_id=0)
            # imageio.imsave(f"yoga_goals/{key}.png", img)

    def compute_reward(self, phy: np.ndarray, g: str) -> float:
        assert g in self._goals.keys()
        distance = _oracle_distance(phy, self._goals[g])
        return - distance


def _shortest_angle(angle):
    if not angle.shape:
        return _shortest_angle(angle[None])[0]
    angle = angle % (2 * np.pi)
    angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
    return angle


def _oracle_distance(x1, x2):
    assert x1.shape[0] in [9, 18], x2.shape[0] in [9, 18]
    x1, x2 = x1[:9], x2[:9]

    def get_su(_goal):
        dist = np.abs(x1 - _goal)
        dist = dist[..., [0, 2, 3, 4, 6, 7]]
        dist[..., 1] = _shortest_angle(dist[..., 1])
        return dist.max(-1)

    return min(get_su(x2), get_su(x2[..., [0, 1, 2, 6, 7, 8, 3, 4, 5]]))


def get_obs_from_yoga_goal(env, goal):
    new_state = np.pad(goal, (0, 9), mode="constant")
    env.physics.set_state(new_state)
    env.physics.forward()
    obs = env.task.get_observation(env.physics)
    return _flatten_obs(obs)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


def get_walkeryoga_goals():
    # pose[0] is height
    # pose[1] is x
    # pose[2] is global rotation
    # pose[3:6] - first leg hip, knee, ankle
    # pose[6:9] - second leg hip, knee, ankle
    # Note: seems like walker can't bend legs backwards

    lie_back = [-1.2, 0., -1.57, 0., 0., 0., 0, -0., 0.]
    lie_front = [-1.2, -0, 1.57, 0., 0, 0., 0., 0., 0.]
    legs_up = [-1.24, 0., -1.57, 1.57, 0., 0.0, 1.57, -0., 0.0]

    kneel = [-0.5, 0., 0., 0., -1.57, -0.8, 1.57, -1.57, 0.0]
    side_angle = [-0.3, 0., 0.9, 0., 0., -0.7, 1.87, -1.07, 0.0]
    stand_up = [-0.15, 0., 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1]

    lean_back = [-0.27, 0., -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]
    boat = [-1.04, 0., -0.8, 1.6, 0., 0.0, 1.6, -0., 0.0]
    bridge = [-1.1, 0., -2.2, -0.3, -1.5, 0., -0.3, -0.8, -0.4]

    head_stand = [-1, 0., -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]
    one_feet = [-0.2, 0., 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]
    arabesque = [-0.34, 0., 1.57, 1.57, 0, 0., 0, -0., 0.]

    return {'lie_back': np.array(lie_back, dtype=np.float32),
            'lie_front': np.array(lie_front, dtype=np.float32),
            'legs_up': np.array(legs_up, dtype=np.float32),
            'kneel': np.array(kneel, dtype=np.float32),
            'side_angle': np.array(side_angle, dtype=np.float32),
            'stand_up': np.array(stand_up, dtype=np.float32),
            'lean_back': np.array(lean_back, dtype=np.float32),
            'boat': np.array(boat, dtype=np.float32),
            'bridge': np.array(bridge, dtype=np.float32),
            'one_feet': np.array(one_feet, dtype=np.float32),
            'head_stand': np.array(head_stand, dtype=np.float32),
            'arabesque': np.array(arabesque, dtype=np.float32)
            }


def extract_names(string: str) -> tp.Set[str]:
    rl = BytesIO(string.encode('utf-8')).readline
    tokens = list(tokenize.tokenize(rl))
    return {t.string for t in tokens if t.type == token.NAME}


class WalkerEquation(BaseReward):

    def __init__(self, string: str) -> None:
        super().__init__()
        self._env = _make_env("walker")
        self._np = ["sin", "cos", "tan", "abs", "exp", "sqrt"]
        variables = list(self._extract(self._env)) + self._np
        not_allowed = extract_names(string) - set(variables)
        # keep this safety measure to avoid being hacked in the demo!
        if not_allowed:
            raise ValueError(f"Following variables are not allowed: {not_allowed}\nPlease only use {variables}")
        self.string = string
        self._precomputed: tp.Dict[str, np.ndarray] = {}

    def _extract(self, env: dmc.EnvWrapper) -> tp.Dict[str, float]:
        data = env.physics.named.data
        return dict(
            x=data.xpos["torso", "x"],
            z=data.xpos["torso", "z"],
            vx=env.physics.horizontal_velocity(),
            vz=env.physics.named.data.sensordata["torso_subtreelinvel"][-1],
            up=env.physics.torso_upright(),
            am=env.physics.named.data.subtree_angmom['torso', 'y']
        )

    def from_env(self, env: dmc.EnvWrapper) -> float:
        # pylint: disable=eval-used
        variables = self._extract(env)
        for name in self._np:
            variables[name] = getattr(np, name)
        return eval(self.string, {}, variables)  # type: ignore

    def _precompute_for_demo(self, workspace: tp.Any) -> None:
        """special method for the demo which precomputes data
        please only use in demo, since it's messy
        """
        ws = workspace
        if hasattr(ws, "_precomputed_"):
            self._precomputed = ws._precomputed_
            return
        import torch  # pylint: disable=import-outside-toplevel
        replay: ReplayBuffer = ws.replay_loader  # recover some typing
        batch = replay.sample(ws.agent.cfg.num_inference_steps, with_physics=True)
        with torch.no_grad():
            obs = torch.Tensor(batch.goal).to(ws.cfg.device)
            B = workspace.agent.backward_net(obs).detach().cpu().numpy()
        precomputed = {"#B": B.astype(np.float32)}
        for k, phy in enumerate(batch._physics):  # type: ignore
            with self._env.physics.reset_context():
                self._env.physics.set_state(phy)
            step_feat = self._extract(self._env)
            for key, val in step_feat.items():
                if key not in precomputed:
                    precomputed[key] = np.zeros((B.shape[0], 1), dtype=np.float32)
                precomputed[key][k] = val
        ws._precomputed_ = precomputed  # store it for reuse
        self._precomputed = precomputed

    def _from_precomputed(self) -> tp.Dict[str, np.ndarray]:
        variables = dict(self._precomputed)
        for name in self._np:
            variables[name] = getattr(np, name)
        rewards = eval(self.string, {}, variables)  # type: ignore
        z = self._precomputed["#B"].T.dot(rewards).squeeze()
        if True:  # ASSUMING SCALED
            z *= np.sqrt(z.size) / np.linalg.norm(z)
        meta = OrderedDict()
        meta['z'] = z
        return meta


class WalkerRandomReward(WalkerEquation):
    """Deterministic positional reward"""

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        rng = np.random.RandomState(seed)
        x = rng.uniform(3, 15)
        nx = rng.uniform(3, 8)
        # equation + weight
        cases = [
            (f"exp(-(x-{x:.1f})**2)", 5),
            (f"exp(-(x-{x:.1f})**2) * up", 5),
            (f"exp(-(x+{nx:.1f})**2)", 2),
            ("vx > 1", 1),
            ("vx > 3", 1),
            ("vx < -1", 1),
        ]
        p = np.array([float(x[1]) for x in cases])
        p /= p.sum()
        selected = cases[rng.choice(range(p.size), p=p)][0]
        super().__init__(selected)
        self._rng = rng
