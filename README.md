[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/controllable_agent/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/controllable_agent/tree/main)

# Controllable Agent

A controllable agent is a reinforcement learning agent whose reward function can be set in real time, without any additional learning or fine-tuning, based on a reward-free pretraining phase.

This project builds a controllable agent based on the Forward-Backward representation from our papers:
- [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935)
- [Learning One Representation to Optimize All Rewards (Neurips 2021)](https://arxiv.org/abs/2103.07945)


## Using the Platform

As an on-going research project, please only expect limited support and no backward-compatibility.


### Structure

The repository is made of 2 packages: `url_benchmark` and `controllable_agent`:
- `controllable_agent` package is only a wrapper around `url_benchmark` to ease experimentation at scale.
- `url_benchmark` package is heavily based on the [`rll-research/url_benchmark`](https://github.com/rll-research/url_benchmark) repository. Main differences include additional agents (notably the Forward-Backward agent), a simplified replay buffer, and structural updates of the package.


### Quick Install

The project dependencies are specificied in the `requirements.txt`, to which one needs to add installation of [`mujoco`](https://github.com/openai/mujoco-py).

You can install the full environment, including mujoco with: \
`source env.sh install <env_name>` (`env_name` defaults to `ca` if not specified)

You can activate the environment with: \
`source env.sh activate <env_name>` (`env_name` defaults to `ca` if not specified)

**Note**: by default, rendering for Mujoco is performed with egl, if that does not work for you, you can try `export MUJOCO_GL=glfw`


### Training

The main entry point is the `url_benchmark.pretrain` command.

#### Commandline

As the command-line interface is provided through [`hydra`](https://github.com/facebookresearch/hydra), you can check available parameters and their default values through:
`python -m url_benchmark.pretrain --cfg job`

In particular, configuration for training with `fb_ddpg` (Forward-Backward) agent can be obtained through:
`python -m url_benchmark.pretrain agent=fb_ddpg --cfg job`

Remove the `--cfg job` parameter to run training, for instance training on quadruped on a simplified goal space with tensorboard and hiplot logging can be performed through:
`python -m url_benchmark.pretrain agent=fb_ddpg task=quadruped_walk goal_space=simplified_quadruped use_tb=1 use_hiplog=1`

**Notes**: 
- Hydra commandline also has [grid-search capacities](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/#internaldocs-banner) as weel as [SLURM cluster support](https://hydra.cc/docs/1.0/plugins/submitit_launcher/#usage) and many other features.
- other APIs exist such as `url_benchmark.anytrain` with similar functionalities.


#### Using hiplot for Monitoring Training
From your device containing the logs, run the following command from the root folder: \
`python -m hiplot url_benchmark.hiplogs.load --port=XXXX`

Then connect to the path that is printed (make sure you have forwarded your port if you don't have the logs locally), and print the folder containing the logs in the text box. The server will parse the folder recursively and plot all train.csv and eval.csv files.



## Demo

A demo is available at [`https://controllable-agent.metademolab.com/`](https://controllable-agent.metademolab.com/) for testing custom rewards on the walker agent.

The demo is based on a replay buffer generated through:
`python -m url_benchmark.anytrain reward_free=1 num_train_episodes=2000 replay_buffer_episodes=2000 agent=fb_ddpg task=walker_walk goal_space=walker_pos_speed_z append_goal_to_observation=1 update_replay_buffer=1 load_replay_buffer=...`
with a replay buffer generated through `rnd`.


### Overview

The agent was trained in the Walker environment. We follow the algorithms outlined in the papers, with a restricted control space of 6 variables: x, z, vx, vz, up, am (horizontal and vertical positions of the torso, horizontal and vertical velocities, cosine of torso angle, angular momentum). We also augment the replay buffer dynamically by following the learned policies with various z parameters.

Again, this is a single agent, it wasn't trained on any of those rewards, and there is no finetuning when the reward function is specified. Based on the reward function, a task parameter is computed via an explicit formula, and a policy is applied using this task parameter.

By varying the reward function, we can train the agent to optimize various combinations of variables, as can be seen below. Multiplicative rewards are the easiest way to mix several constraints. 

Rewards must be provided as a Python equation. Here are a few reward examples:
- `vx`: run as fast as possible
- `x < -4`: go to the left until x<-4
- `1 / z`: be close to the ground
- `-up`: be upside down
- `-up * x * (x > 0)`: be to the right and upside down
- `exp(-abs(x - 8)) * up / z`: be around x=8, upright, and close to the ground: crouch at x=8
- `exp(-abs(x - 10)) * up * z**4`: be around x=10, upright, and very high: jump at x=10
- `vx/z**2`: crawl
- `exp(-abs(vx - 2)) * up`: move slowly (speed=2) and stay upright
- `vx * (1 - up) / z`: move as fast as possible, upside down, close to the ground
- `-am * exp (-abs(x - 10))`: go to x=10 and do backward spins
- `vx * (1 + up * cos(x / 4))`: run upright or rolling depending on cos(x/4)


### Running the Demo Locally

Provided you have access to a replay buffer and model checkpoint:

1. Create/activate your environment
2. Update the `CASES` variable to have one entry pointing to your checkpoint
3. From the root of the repo, run `streamlit run demo/main.py --server.port=8501`
4. Connect instead to `localhost:8501` in your browser (don't forget port forwarding if the demo runs on a server)
5. write a formula to be maximized as Python code.


## Contributing 

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
`controllable_agent` is MIT licensed, as found in the LICENSE file.