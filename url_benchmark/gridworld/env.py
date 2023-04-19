# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import enum
import dm_env
from dm_env import specs
import numpy as np
import matplotlib.pyplot as plt
from url_benchmark.dmc import ExtendedTimeStep


class ObservationType(enum.IntEnum):
    STATE_INDEX = enum.auto()
    AGENT_ONEHOT = enum.auto()
    GRID = enum.auto()
    AGENT_GOAL_POS = enum.auto()
    AGENT_POS = enum.auto()


def build_gridworld_task(task,
                         discount=1.0,
                         penalty_for_walls=0,
                         observation_type=ObservationType.AGENT_POS,
                         max_episode_length=200):
    """Construct a particular Gridworld layout with start/goal states.

    Args:
      task: string name of the task to use. One of {'simple', 'obstacle',
        'random_goal'}.
      discount: Discounting factor included in all Timesteps.
      penalty_for_walls: Reward added when hitting a wall (should be negative).
      observation_type: Enum observation type to use. One of:
        * ObservationType.STATE_INDEX: int32 index of agent occupied tile.
        * ObservationType.AGENT_ONEHOT: NxN float32 grid, with a 1 where the
          agent is and 0 elsewhere.
        * ObservationType.GRID: NxNx3 float32 grid of feature channels.
          First channel contains walls (1 if wall, 0 otherwise), second the
          agent position (1 if agent, 0 otherwise) and third goal position
          (1 if goal, 0 otherwise)
        * ObservationType.AGENT_GOAL_POS: float32 tuple with
          (agent_y, agent_x, goal_y, goal_x).
      max_episode_length: If set, will terminate an episode after this many
        steps.
    """
    tasks_specifications = {
        'simple': {
            'layout': [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          ],
          'start_state': (2, 2),
          'randomize_goals': True
          # 'goal_state': (7, 2)

      },
      'obstacle': {
          'layout': [
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, 0, 0, 0, 0, -1, 0, 0, -1],
              [-1, 0, 0, 0, -1, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          ],
          'start_state': (2, 2),
          'goal_state': (2, 8)
      },
      'random_goal': {
          'layout': [
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          ],
          'start_state': (2, 2),
          # 'randomize_goals': True
      },
    }
    return GridWorld(
        discount=discount,
        penalty_for_walls=penalty_for_walls,
        observation_type=observation_type,
        max_episode_length=max_episode_length,
        **tasks_specifications[task])

class GridWorld(dm_env.Environment):

    def __init__(self,
               layout,
               start_state,
               goal_state=None,
               observation_type=ObservationType.STATE_INDEX,
               discount=1.0,
               penalty_for_walls=0,
               reward_goal=1,
               max_episode_length=None,
               randomize_goals=False) -> None:
        """Build a grid environment.

        Simple gridworld defined by a map layout, a start and a goal state.

        Layout should be a NxN grid, containing:
          * 0: empty
          * -1: wall
          * Any other positive value: value indicates reward; episode will terminate

        Args:
          layout: NxN array of numbers, indicating the layout of the environment.
          start_state: Tuple (y, x) of starting location.
          goal_state: Optional tuple (y, x) of goal location. Will be randomly
            sampled once if None.
          observation_type: Enum observation type to use. One of:
            * ObservationType.STATE_INDEX: int32 index of agent occupied tile.
            * ObservationType.AGENT_ONEHOT: NxN float32 grid, with a 1 where the
              agent is and 0 elsewhere.
            * ObservationType.GRID: NxNx3 float32 grid of feature channels.
              First channel contains walls (1 if wall, 0 otherwise), second the
              agent position (1 if agent, 0 otherwise) and third goal position
              (1 if goal, 0 otherwise)
            * ObservationType.AGENT_GOAL_POS: float32 tuple with
              (agent_y, agent_x, goal_y, goal_x)
          discount: Discounting factor included in all Timesteps.
          penalty_for_walls: Reward added when hitting a wall (should be negative).
          reward_goal: Reward added when finding the goal (should be positive).
          max_episode_length: If set, will terminate an episode after this many
            steps.
          randomize_goals: If true, randomize goal at every episode.
        """
        if observation_type not in ObservationType:
            raise ValueError('observation_type should be a ObservationType instace.')
        self._layout = np.array(layout)
        self._start_state = start_state
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount
        self._penalty_for_walls = penalty_for_walls
        self._reward_goal = reward_goal
        self._observation_type = observation_type
        self._layout_dims = self._layout.shape
        self._max_episode_length = max_episode_length
        self._num_episode_steps = 0
        self._randomize_goals = randomize_goals
        self._goal_state: tp.Tuple[int, int]
        if goal_state is None:
            # Randomly sample goal_state if not provided
            goal_state = self._sample_goal()
        self.goal_state = goal_state

    def _sample_goal(self):
        """Randomly sample reachable non-starting state."""
        # Sample a new goal
        n = 0
        max_tries = 1e5
        while n < max_tries:
            goal_state = tuple(np.random.randint(d) for d in self._layout_dims)
            if goal_state != self._state and self._layout[goal_state] == 0:
                # Reachable state found!
                return goal_state
            n += 1
        raise ValueError('Failed to sample a goal state.')

    @property
    def number_of_states(self):
        return self._number_of_states

    @property
    def goal_state(self):
        return self._goal_state

    def set_state(self, x, y):
        self._state = (y, x)

    @goal_state.setter
    def goal_state(self, new_goal):
        if new_goal == self._state or self._layout[new_goal] < 0:
            raise ValueError('This is not a valid goal!')
        # Zero out any other goal
        self._layout[self._layout > 0] = 0
        # Setup new goal location
        self._layout[new_goal] = self._reward_goal
        self._goal_state = new_goal

    def observation_spec(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            return specs.Array(
                shape=(self._number_of_states, ),
                dtype=np.float32,
                name='observation_agent_onehot')
        elif self._observation_type is ObservationType.GRID:
            return specs.Array(
                shape=self._layout_dims + (3,),
                dtype=np.float32,
                name='observation_grid')
        elif self._observation_type is ObservationType.AGENT_POS:
            return specs.Array(
                shape=(2,), dtype=np.float32, name='observation_agent_pos')
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return specs.Array(
                shape=(4,), dtype=np.float32, name='observation_agent_goal_pos')
        elif self._observation_type is ObservationType.STATE_INDEX:
            return specs.DiscreteArray(
                self._number_of_states, dtype=int, name='observation_state_index')

    def action_spec(self):
        return specs.DiscreteArray(5, dtype=int, name='action')

    def get_state(self):
        return self._state

    def get_goal_obs(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            obs = np.zeros(self._layout.shape, dtype=np.float32)
            # Place agent
            obs[self._goal_state] = 1
            return obs.flatten()
        elif self._observation_type is ObservationType.AGENT_POS:
            return np.array(self._goal_state, dtype=np.float32) / np.array(self._layout.shape, dtype=np.float32)
        elif self._observation_type is ObservationType.STATE_INDEX:
            y, x = self._goal_state
            return y * self._layout.shape[1] + x

    def get_obs(self):
        if self._observation_type is ObservationType.AGENT_ONEHOT:
            obs = np.zeros(self._layout.shape, dtype=np.float32)
            # Place agent
            obs[self._state] = 1
            return obs.flatten()
        elif self._observation_type is ObservationType.GRID:
            obs = np.zeros(self._layout.shape + (3,), dtype=np.float32)
            obs[..., 0] = self._layout < 0
            obs[self._state[0], self._state[1], 1] = 1
            obs[self._goal_state[0], self._goal_state[1], 2] = 1
            return obs
        elif self._observation_type is ObservationType.AGENT_POS:
            return np.array(self._state, dtype=np.float32) / np.array(self._layout.shape, dtype=np.float32)
        elif self._observation_type is ObservationType.AGENT_GOAL_POS:
            return np.array(self._state + self._goal_state, dtype=np.float32)
        elif self._observation_type is ObservationType.STATE_INDEX:
            y, x = self._state
            return y * self._layout.shape[1] + x

    def reset(self):
        self._state = self._start_state
        self._num_episode_steps = 0
        if self._randomize_goals:
            self.goal_state = self._sample_goal()
        return ExtendedTimeStep(
            step_type=dm_env.StepType.FIRST,
            action=0,
            reward=0.0,
            discount=1,
            observation=self.get_obs())

    def step(self, action):
        y, x = self._state
        if action == 0:  # up
          new_state = (y - 1, x)
        elif action == 1:  # right
          new_state = (y, x + 1)
        elif action == 2:  # down
          new_state = (y + 1, x)
        elif action == 3:  # left
          new_state = (y, x - 1)
        elif action == 4: # stay
          new_state = (y, x)
        else:
          raise ValueError(
              'Invalid action: {} is not 0, 1, 2, 3, or 4.'.format(action))

        new_y, new_x = new_state
        step_type = dm_env.StepType.MID
        if self._layout[new_y, new_x] == -1:  # wall
            reward = self._penalty_for_walls
            discount = self._discount
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = 0.
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_y, new_x]
            ##  if we choose de terminate
            # discount = 0.
            # new_state = self._start_state
            # step_type = dm_env.StepType.LAST
            discount = self._discount

        self._state = new_state
        self._num_episode_steps += 1
        if (self._max_episode_length is not None and
            self._num_episode_steps >= self._max_episode_length):
          step_type = dm_env.StepType.LAST
        return ExtendedTimeStep(
            step_type=step_type,
            action=action,
            reward=np.float32(reward),
            discount=discount,
            observation=self.get_obs())

    def plot_grid(self, add_start=True):
        asbestos = (127 / 255, 140 / 255, 141 / 255, 0.8)
        dodger_blue = (25 / 255, 140 / 255, 255 / 255, 0.8)
        # carrot = (235 / 255, 137 / 255, 33 / 255, 0.8)
        grid_kwargs = {'color': (220 / 255, 220 / 255, 220 / 255, 0.5)}
        # marker_style = dict(linestyle=':', color=carrot, markersize=20)
        plt.figure(figsize=(4, 4))
        img = np.ones((self._layout.shape[0], self._layout.shape[1], 4))
        wall_y, wall_x = np.where(self._layout <= -1)
        for i in range(len(wall_y)):
            img[wall_y[i], wall_x[i]] = np.array(asbestos)

        plt.imshow(img, interpolation=None)
        # plt.imshow(self._layout <= -1, interpolation='nearest')
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        # Add start/goal
        if add_start:
            plt.text(
            self._start_state[1],
            self._start_state[0],
            r'$\mathbf{S}$',
            fontsize=16,
            ha='center',
            va='center')
        plt.text(
            self._goal_state[1],
            self._goal_state[0],
            r'$\mathbf{G}$',
            fontsize=16,
            ha='center',
            va='center',
            color=dodger_blue)
        h, w = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], **grid_kwargs)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], **grid_kwargs)

    def render(self, return_rgb=True):
        carrot = (235 / 255, 137 / 255, 33 / 255, 0.8)
        self.plot_grid(add_start=False)
        # Add the agent location
        plt.text(
            self._state[1],
            self._state[0],
            u'😃',
            # fontname='symbola',
            fontsize=18,
            ha='center',
            va='center',
            color=carrot)
        if return_rgb:
            fig = plt.gcf()
            plt.axis('tight')
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            plt.close(fig)
            return data

    def plot_policy(self, policy):
        action_names = [
            r'$\uparrow$', r'$\rightarrow$', r'$\downarrow$', r'$\leftarrow$'
        ]
        self.plot_grid()
        plt.title('Policy Visualization')
        h, w = self._layout.shape
        for y in range(h):
            for x in range(w):
                # if ((y, x) != self._start_state) and ((y, x) != self._goal_state):
                if (y, x) != self._goal_state:
                    action_name = action_names[policy[y, x]]
                    plt.text(x, y, action_name, ha='center', va='center')

    def plot_greedy_policy(self, q):
        greedy_actions = np.argmax(q, axis=2)
        self.plot_policy(greedy_actions)
