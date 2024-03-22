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

    # config.augmentation_data_files = ["augment_dataset_aug-10_m-policy_variance.pkl"]
    return config
