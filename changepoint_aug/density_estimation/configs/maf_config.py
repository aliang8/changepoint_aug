from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.results_file = "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/model_ckpts/maf_params.pkl"
    config.vizdom_name = "maf_2"
    return config
