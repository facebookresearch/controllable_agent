# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful links:
Streamlit cheatsheet:
https://docs.streamlit.io/library/cheatsheet

Also check the components we provide for demos in metastreamlit:
https://github.com/fairinternal/metastreamlit
You can request new components by creating an issue
"""

# Designed to run from controllable_agent with streamlit run demo/main.py
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # avoid using CUDA
import sys
import time
import tempfile
from pathlib import Path
from collections import OrderedDict
import streamlit as st
try:
    import url_benchmark
    base = Path(url_benchmark.__file__).absolute().parents[1]
except ImportError:
    base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import url_benchmark
# we need to add url_benchmar to be able to reload legacy checkpoints
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
print("base", base)
from url_benchmark import pretrain
import numpy as np
import torch
import torch.nn.functional as F
from controllable_agent import runner
from url_benchmark import goals
from url_benchmark import utils
from url_benchmark.video import VideoRecorder

st.set_page_config(
    page_title="Controllable agent - Meta AI",
    menu_items={"About": "This demo is powered by the code available at https://github.com/facebookresearch/controllable_agent\nCopyright 2022 Meta Inc. Available under MIT Licence."},

)
# st.title('Controllable agent')
st.sidebar.write('# Controllable Agent Demo')
st.sidebar.write("### Optimize Any Reward Function with a Single Pretrained Agent")
st.sidebar.write("***Ahmed Touati, Jérémy Rapin, Yann Ollivier***")
st.sidebar.write("A controllable agent is a reinforcement learning agent whose _reward function can be set in real time_, without any additional learning or fine-tuning, based on a reward-free pretraining phase.")
st.sidebar.write("""The controllable agent here uses the _forward-backward representation_ from our papers:
* [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935)
* [Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945) (Neurips 2021)
""")
st.sidebar.write("The [code is open-source](https://github.com/facebookresearch/controllable_agent).")

model_path = Path("/checkpoint/jrapin/ca/models")
if not model_path.exists():
    model_path = base / "models"

# having more cases will trigger a dropdown box
CASES = {
    # Update the following path to a checkpoint that exists in you system
    "walker - 221020 (rnd init)": model_path / "walker_rnd_init_65697627_11_221020.pt",
}
CASES = {x: y for x, y in CASES.items() if y.exists()}
if len(CASES) > 1:
    case = st.selectbox(
        'Which model do you want to load?',
        list(CASES)
    )
else:
    case = list(CASES)[0]
assert case is not None


@st.cache(max_entries=1, allow_output_mutation=True)
def load_workspace(case: str):
    checkpoint = CASES[case]
    hp = runner.HydraEntryPoint(base / "url_benchmark/anytrain.py")
    ws = hp.workspace(task="walker_walk", replay_buffer_episodes=2000, goal_space="walker_pos_speed_z", append_goal_to_observation=True)
    ws.train_env.reset()
    with checkpoint.open("rb") as f:
        payload = torch.load(f, map_location=ws.device)
    ws.agent = payload["agent"]
    ws.agent.cfg.device = ws.cfg.device
    replay = payload["replay_loader"]
    ws.replay_loader = replay
    ws.replay_storage = replay
    return ws


# load
ws = load_workspace(case)
recorder = VideoRecorder(base, camera_id=ws.video_recorder.camera_id, use_wandb=False)
recorder.enabled = True


reward = goals.WalkerEquation("x")
reward._precompute_for_demo(ws)  # precompute before first run
params = list(reward._extract(reward._env))
params_str = ", ".join(f"`{x}`" for x in params)

st.write("##### Try Your Own Reward Function for Walker")
st.write(f"Enter a Walker reward function to maximize, such as `-vx` or `exp(-(x-8)**2)`\n\n This can be any Python equation using {params_str} (horizontal and vertical position, horizontal and vertical speed, sine of torso angle, angular momentum)")
string = st.text_input("Reward function:", value=st.session_state.get("prefill", ""))
# st.session_state.pop("prefill", None)

col1, col2 = st.columns(2)

early_stopping = True
last_physics = np.ndarray([])
if string and string is not None:
    reward = goals.WalkerEquation(string)
    reward._precompute_for_demo(ws)  # loads from cached workspace if already precomputed
    print(f"Running reward: {string}")  # for the console
    col1.write(f"Running reward `{string}`")  # use code formating to avoid italic from **
    if not reward._precomputed:
        meta = pretrain._init_eval_meta(ws, custom_reward=reward)
    else:
        print("Inferring from precomputed data")
        meta = reward._from_precomputed()

    col1.write("Applying the policy for 500 time steps and generating video (this may take 10-15s)")
    # play
    env = ws._make_env()
    time_step = env.reset()
    recorder.init(env)
    total_reward = 0
    k = 0
    durations = dict(model=0.0, env=0.0, render=0.0)
    t_start = time.time()
    while k < 500 and not time_step.last():
        k += 1
        t0 = time.time()
        with torch.no_grad(), utils.eval_mode(ws.agent):
            action = ws.agent.act(time_step.observation,
                                  meta,
                                  1000000,
                                  eval_mode=True)
        t1 = time.time()
        time_step = env.step(action)
        t2 = time.time()
        recorder.record(env)
        t3 = time.time()
        durations["model"] += t1 - t0
        durations["env"] += t2 - t1
        durations["render"] += t3 - t2
        total_reward += reward.from_env(env)
        distance = np.linalg.norm(time_step.physics - last_physics) / time_step.physics.size
        if early_stopping and distance < 5e-6:
            print(f"Early stopping at time step {k}")
            break
        last_physics = time_step.physics
    print(f"Total play time {time.time() - t_start:.2f}s with {durations}")
    state = reward._extract(env)
    state_str = " ".join(f"{x}={y:.2f}" for x, y in state.items())
    col1.write(
        f"Average reward is {total_reward / k}\n\n"
        f'Final state is {state_str}'
    )
    name = "demo.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        recorder.save_dir = Path(tmp)
        t0 = time.time()
        recorder.save(name)
        print(f"Saved video to {recorder.save_dir / name} in {time.time() - t0:.2f}s, now serving it.")
        col = st.columns([1, 3, 1])[1]
        with col:
            col2.video(str(recorder.save_dir / name))

st.write("---")
st.write(f"""**Note**: multiplicative rewards are a good way to combine constraints on the agent. For instance, `z**4 * exp(-abs(x-5))` makes the agent try to jump around `x=5`""")
st.write(f"""This agent is far from perfect, and it is still easy to make it fail. For instance, the variable `vz` does not seem to do much. `x**2` produces bad results, presumably because rewards are largest at places that were far from the replay buffer. On the other hand, `x**2 * (x<20) * (x>-20)` works better, because the reward is restricted to a well-explored zone.""")

with st.expander("How Does This Work?"):
    st.write("""
The algorithms are directly taken from our papers (see side bar). At pre-training time, two representations $F(s,a,z)$ and $B(s)$ ("forward" and "backward") were learned, as well as a parametric policy $\pi_z(s)$. Here $z$ is a hidden variable in representation space.

When a new reward function $r$ is set, the app computes the hidden variable $z=\mathbb{E}[r(s)B(s)]$ using 5,000 states $s$ from the training set, using the provided function $r$. Then the policy $\pi_z$ with parameter $z$ is deployed.

The dimension of $F$, $B$ and $z$ is 50. The networks are small multilayer perceptrons. The training set was initialized by a standard exploration algorithm, Random Network Distillation. It is made of 2,000 length-1,000 trajectories. Then we learn $F$, $B$ and $\pi_z$ using the method described in our papers, and we update the training set by sampling random $z$ and applying the corresponding policy. 

For $B$, we only provide a subset of variables from the full state $s$, namely, the six variables `x,z,vx,vz,up,am` mentioned above, to focus training on those. Our theory guarantees that, if the networks minimize the loss well, all reward functions depending on those variables will be optimized.

###### How do we Learn $F$, $B$ and $\pi$? Causes and Effects

Intuitively, $F(s,a,z)$ represents the "effects" of following $\pi_z$ starting at state-action $(s,a)$, while $B(s')$ represents the possible "causes" leading to state $s'$.

If it's easy to reach $s'$ while starting at $s,a$ and following $\pi_z$ for many steps, then the dot product $F(s,a,z)^TB(s')$ will be large, meaning, we align the representation vectors $F(s,a,z)$ and $B(s')$. The precise equation (below) uses the cumulated long-term transition probabilities between states.

The policy $\pi_z$ is trained to return an action $a$ that maximizes $F(s,a,z)^T z$.

The full set of equations is:""")
    st.latex(r'''\begin{cases}
    \pi_z(s)=\mathrm{argmax}_a \, F(s,a,z)^T z\\
    F(s,a,z)^T B(s') \rho(s') = \sum_t \gamma^t \Pr(s_t=s'|s_0=s,a_0=a,\pi_z)
    \end{cases}
''')
    st.write("""
Here $\\rho$ is the distribution of states in the training set (we don't need to know $\\rho$, just to sample from it).

Our theory guarantees this provides all optimal policies if training is successful:

**Theorem.** *Assume the equations above hold. Then the optimal policy for any reward function $r$ can be obtained by evaluating* """)
    st.latex(r''' z=\mathbb{E}[r(s)B(s)] ''')
    st.write(""" *on states sampled from the training distribution, and applying policy $\pi_z$.*

*Moreover, approximate solutions still provide approximately optimal policies.*

The equation on $F$ and $B$ seems hard to handle, but it can be rewritten as a kind of generalized Bellman equation for $F^T B$, which we use for training. There is no representation collapse ($F=B=0$ does not satisfy the equation). There is no sparse reward problem from $\Pr(s_t=s')$, thanks to our probability-measure-valued treatment of the equation.

Overall, this is somewhat similar to a world model except:
* There is no planning at test time
* We never synthesize states or imaginary trajectories
* We learn long-term transition probabilities for many policies instead of one-step, policy-independent next states

""")


st.write("##### Some Examples")
reward_texts = [
    ("vx", "run as fast as possible"),
    ("x < -4", "go to the left until x<-4"),
    ("1 / z", "be close to the ground"),
    ("-up", "be upside down"),
    ("-up * x * (x > 0)", "be to the right and upside down"),
    ("(1-up) * exp(-abs(x-10))", "be upside down around x=10"),
    ("exp(-abs(x - 8)) * up / z", "be around x=8, upright, and close to the ground: crouch at x=8"),
    ("exp(-abs(x - 10)) * up * z**4", "be around x=10, upright, and very high: jump at x=10"),
    ("vx/z**2", "crawl"),
    ("exp(-abs(vx - 2)) * up", "move slowly (speed=2) and stay upright"),
    ("vx * (1 - up) / z", "move as fast as possible, upside down, close to the ground"),
    ("vx * (1 + up * cos(x / 4))", "run upright or rolling depending on cos(x/4)"),
]


def _prefill(eq: str) -> None:
    st.session_state["prefill"] = eq


for reward, text in reward_texts:
    cols = st.columns(3)
    cols[0].write(f"`{reward}`")
    cols[1].write(text)
    cols[2].button("Try", key=reward, on_click=_prefill, args=(reward,))
    # col[2].write("video TODO")
