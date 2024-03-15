from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "bc"
    config.exp_name = "bc"
    config.vizdom_name = "bc"

    # policies
    config.num_policies = 5
    config.policy_cls = "mlp"
    config.num_trajs = 200
    return config
