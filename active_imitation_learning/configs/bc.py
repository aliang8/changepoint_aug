from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "bc"
    config.exp_name = "bc"
    # save best model for eval based on this metric
    config.save_key = "rollout/avg_success_rate"
    config.best_metric = "max"

    # policies
    config.num_policies = 5
    config.policy_cls = "mlp"
    config.num_trajs = 200

    config.save_interval = 100
    config.test_interval = 100
    config.num_epochs = 1000
    config.num_eval_episodes = 50

    # augmentation parameters
    config.num_augmentation_steps = 1000
    # config.augmentation_data_files = [
    #     "d_eid-MAZE_nes-10_nap-1_nps-1_selm-policy_variance_sel-per_traj_lam-1.3_rwd-False_tns-100"
    # ]
    # config.augmentation_data_files = [
    #     "d_eid-MAZE_nes-20_nap-1_nps-2_selm-influence_function_sel-per_traj_lam-1.3_rwd-False_tns-100"
    # ]
    config.augmentation_data_files = [
        "d_eid-MAZE_nes-20_nap-1_nps-1_selm-policy_variance_sel-global_lam-1.3_rwd-False_tns-100_rand-True"
    ]
    return config
