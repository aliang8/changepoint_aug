from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "bc"
    config.exp_name = "bc"
    # save best model for eval based on this metric
    config.save_key = "ic_rollouts/avg_success_rate"
    config.best_metric = "max"
    config.env_id = "maze_2d_wall_v0"

    # policies
    config.num_policies = 5
    config.policy_cls = "gaussian"
    config.num_successes_save = 5
    config.num_failures_save = 5
    config.num_videos_save = 5

    config.save_interval = 200
    config.test_interval = 200
    config.num_epochs = 5000
    config.train_perc = 1.0

    config.num_eval_episodes = 50
    config.max_episode_steps = 250

    # augmentation parameters
    config.augmentation_data = ()
    config.num_augmentation_steps = -1
    config.base_num_trajs = 5
    config.num_additional_trajs = 0
    config.num_shuffles = 2
    config.perturbations_per_trajs = 2
    config.run_ic_evals = False

    # env specific things
    config.clip_action = 0.999
    config.reward_scale = 1.0
    config.reward_bias = 0.0

    config.keys_to_include = {
        "num_policies": None,
        "num_epochs": None,
        "base_num_trajs": None,
        "num_additional_trajs": None,
        "policy_cls": None,
        "trainer": None,
        "env_id": None,
        "num_augmentation_steps": None,
    }
    return config
