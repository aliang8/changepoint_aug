from configs.augment import get_config as get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.exp_name = "augment"

    config.env = "MW"
    config.env_id = "drawer-open-v2"
    config.data_file = "sac_mw_drawer-open-v2_100.pkl"

    # for loading model to compute the density
    config.density_exp_name = "i006_cvae"
    config.density_model_ckpt = "nt-200"
    config.density_ckpt_step = 950

    # for loading model to compute the heuristic
    config.exp_name = "i002_q_sarsa_mw"
    config.model_ckpt = "nt-100_s-0"
    # config.exp_name = "i014_bc_mw"
    # config.model_ckpt = "nt-50_s-0"
    config.ckpt_step = 900

    config.oracle_model_ckpt_path = "/scr/aliang80/changepoint_aug/changepoint_aug/old/model_ckpts/mw_drawer-open-v2_200000.pt"

    config.augment_data_file = ""
    return config
