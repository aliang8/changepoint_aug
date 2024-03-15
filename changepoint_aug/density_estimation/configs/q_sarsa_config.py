from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "q_sarsa"
    config.exp_name = "q_sarsa"
    config.vizdom_name = "q_2"

    # qfunction
    config.gamma = 0.99
    config.tau = 0.005
    return config
