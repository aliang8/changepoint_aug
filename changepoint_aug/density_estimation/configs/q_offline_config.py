from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.results_file = "q_params.pkl"
    config.vizdom_name = "q_2"

    # policy
    config.num_ensemble = 1

    # qfunction
    config.gamma = 0.99
    config.tau = 0.005
    config.target_update_freq = 2
    return config
