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

    config.num_eval_episodes = 50

    # config.augmentation_data_files = [
    #     "augment_dataset_lam-1.3_nap-1_nes-20_nps-2_rwd-True_sel-per_traj_selm-policy_variance_tns-100.pkl"
    # ]
    config.augmentation_data_files = [
        "augment_dataset_lam-1.3_nap-2_nes-50_nps-2_rwd-False_sel-per_traj_selm-influence_function_tns-100.pkl"
    ]

    return config
