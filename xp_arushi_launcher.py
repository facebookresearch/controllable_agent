import datetime
import submitit
from controllable_agent import runner
from itertools import product

# experiment = "new_explorer_agent"
# experiment = "exp_T_mlt_explorer_agent"
# experiment = "cov_cr_explorer_agent"
# experiment = "cov_cr_5K_explorer_agent"

# experiment = "fb_loss_explorer_agent"
# experiment = "new_agent"
# experiment = "maze_agent"

# experiment = "mlt_goals_walker_new_rp_2"
# experiment = "mlt_goals_maze"
# experiment = "grid_online_new"
# experiment = "diffusion_new"
# experiment = "mean_z"
# experiment = "grid_o"

# experiment = "noisy_obs"
# experiment = "FB_w_wo_mix_z"
# experiment = "FB_worst_z_mix_z" # worst z with and without mix z in optim level only
# experiment = "mix_z_ratio"
# experiment = "worst_z_exp_opt"
# experiment = "new_noisy" # 30 dim noise vector, 20 mean, 5 std
# experiment = "pure_exp"
# experiment = "weighted_loss"
experiment = "samples_for_eval_z_1"

z_dim = 50
num_eval_episodes=5
eval_every_episodes = 200

basic_maze = {
    "experiment": experiment,
    "task": "point_mass_maze_reach_top_right",
    "goal_space" : "simplified_point_mass_maze",
    "agent.z_dim": z_dim,
    "agent.batch_size": 1024,
    "agent.q_loss": 1,
    "agent.lr": 1e-4,
    "discount": 0.99,
    "agent.backward_hidden_dim": 256,
    "agent.add_trunk": 1,
    "seed": 1,
    "reward_free": 1,
    "final_tests": 10,  # EVALUATION
    "future": 0.99,
    "custom_reward_maze": "maze_multi_goal",
    # added exploration stuff
    "num_train_episodes": 3000,
    "num_rollout_episodes": 10,
    "agent.exploration_size": 8000,
    "num_eval_episodes": num_eval_episodes,
    "explore": 1,
    "agent.explore_ratio": 0,
    "num_seed_frames": 4000,
    "checkpoint_every": 1000, # checkpoint every episodes
    "eval_every_episodes": eval_every_episodes,
    "num_seed_episodes": 10, # seed episodes for which agent is not updated
    "num_seed_episodes_explore":30,
    "num_agent_updates": 2500,
    "agent.update_explore_z_after_every_step": 300,
    "agent.ratio_data_explore": 0.3,
    "wandb_end_name": experiment,
    "update_target_net_prob": 0.5,
    "agent.ortho_coef": 0.1,
    "agent.linearized_z_exploration_technique": True,
    "agent.L2_norm_z_explore": True,
    # separate agent for exploration
    "update_cov_steps": 500,
    "update_exploratory_agent_every_steps": 5,
    "separate_exploratory_agent": True,
    "exp_cov_F": False,
    "exp_cov_B": False,
    "exp_cov_FB": False,
    "intr_reward_FBloss": False,
    "ema_intr_reward": False,
    "explore_rnd": True
}

basic_grid = {
    "task": "grid_simple",
    "experiment": experiment,
    "discount": 0.99,
    "agent.batch_size": 1024,
    "agent.q_loss": 1,
    "agent.q_loss_coef": 0.01,
    "agent.backward_hidden_dim": 256,
    "future": 0.99,
    "agent.lr": 1e-4,
    "seed": 1,
    "num_seed_frames": 4000,
    "checkpoint_every": 500, # checkpoint every episodes
    "eval_every_episodes": 20,
    # change them below here
    "num_seed_episodes": 10, # seed episodes for which agent is not updated
    "agent.z_dim": z_dim,
    "num_train_episodes": 1000,
    "num_rollout_episodes": 10,
    "agent.exploration_size": 8000,
    "num_eval_episodes": 15,
    "explore": 0,
    "agent.explore_ratio": 0,
    "num_seed_episodes_explore":5,
    "num_agent_updates": 2500,
    "agent.update_explore_z_after_every_step": 300,
    "agent.ratio_data_explore": 0.3,
    "wandb_end_name": experiment,
    "update_target_net_prob": 0.5,
    "agent.ortho_coef": 0.1,
    "agent.linearized_z_exploration_technique": True,
    "agent.L2_norm_z_explore": True,
    "set_epsilon_exploration": False,
    "epsilon_explore_val": 0.8,
    # separate agent for exploration
    "update_cov_steps": 500,
    "update_exploratory_agent_every_steps": 5,
    "exp_cov_F": False,
    "exp_cov_B": False,
    "exp_cov_FB": False,
    "intr_reward_FBloss": False,
    "ema_intr_reward": False,
    "separate_exploratory_agent": False,
    "explore_rnd": False,
    "reward_prioritization": False,
    "agent.compute_z_from_FB_reward": False
}


basic_walker = {
    "task": "walker_stand",
    "experiment": experiment,
    "discount": 0.98,
    "agent.batch_size": 1024,
    "agent.q_loss": 1,
    "agent.backward_hidden_dim": 256,
    "agent.add_trunk": 0,
    "future": 0.99,
    "agent.lr": 1e-4,
    "seed": 1,
    "num_seed_frames": 4000,
    "checkpoint_every": 2000, # checkpoint every episodes
    "eval_every_episodes": eval_every_episodes,
    # change them below here
    "num_seed_episodes": 10, # seed episodes for which agent is not updated
    "agent.z_dim": z_dim,
    "agent.mix_ratio": 0.5,
    "num_train_episodes": 5000,
    "num_rollout_episodes": 10,
    "agent.exploration_size": 8000,
    "num_eval_episodes": num_eval_episodes,
    "explore": 0,
    "agent.explore_ratio": 0,
    "num_seed_episodes_explore":30,
    "num_agent_updates": 5000,
    "agent.update_explore_z_after_every_step": 300,
    "agent.ratio_data_explore": 0.3,
    "wandb_end_name": experiment,
    "update_target_net_prob": 0.5,
    "agent.ortho_coef": 0.1,
    "agent.linearized_z_exploration_technique": True,
    "agent.L2_norm_z_explore": True,
    "set_epsilon_exploration": False,
    "epsilon_explore_val": 0.8,
    # separate agent for exploration
    "update_cov_steps": 500,
    "update_exploratory_agent_every_steps": 5,
    "custom_reward_mujoco": "mujoco_tasks",
    "exp_cov_F": False,
    "exp_cov_B": False,
    "exp_cov_FB": False,
    "ema_intr_reward": False,
    "explore_rnd": 0,
    "reward_prioritization": False,
    "agent.compute_z_from_FB_reward": False,
    "uniform_explore_after_few_episodes": False,
    "exploratory_diffusion_after_every_episode": 500,
    "exploration_in_last_steps": 0.9,
    "intr_reward_FBloss": 1,
    "separate_exploratory_agent": True,
    "compute_worst_z_FB_loss": 0,
    "add_noise_to_obs": 0,
    "type_noise": "sin",
    # PAL loss function for FB optimization
    "agent.fb_weighted_loss": False,
    "agent.alpha": 0.4,
    "agent.min_priority": 1

}

algos = [ {"agent": "fb_ddpg"}]
# algos = [ {"agent": "discrete_fb"}]
variations = {
    "explore": [1],
    "seed": [1, 2],
    "agent.num_inference_steps": [1000, 10000, 30000, 50000]
}


# variations = {
#     "explore": [0],
#     "seed": [1],
#     "agent.ortho_coef": [0.1, 1],
#     "num_rollout_episodes": [5],
#     "num_agent_updates": [500, 100],
#     "agent.z_dim": [100],
#     "agent.q_loss_coef": [0.01]
# }

basic = basic_walker #basic_grid

experiment = basic["experiment"]
date = datetime.date.today().isoformat()
executor = submitit.AutoExecutor(folder=f"/checkpoint/arushijain/ca/{date}_{experiment}/slurm/%j")
executor.update_parameters(timeout_min=70 * 60, slurm_partition="learnlab", slurm_array_parallelism=512,
                           gpus_per_node=1, cpus_per_task=6, name=experiment)
xpfolder = executor.folder.parents[1]
hp = runner.CopiedBenchmark(xpfolder, "train_online")
print(f"Xp folder: {xpfolder}")


with executor.batch():
    keys, vals = zip(*variations.items())
    print(vals)
    for bundle in product(*vals):
        new_variation = dict(zip(keys, bundle))
        for algo in algos:
            name = algo["agent"]
            work_dirname = ""
            params = dict(basic)
            params.update(algo)
            for key, val in new_variation.items():
                work_dirname+= key + "_" + str(val)
                params.update(**{key: val})
            print("work dir name:", work_dirname)
            print("final params:")
            print(params)
            work_dir = xpfolder / "results" / name / work_dirname
            job = executor.submit(hp, _working_directory_=work_dir, **params)
