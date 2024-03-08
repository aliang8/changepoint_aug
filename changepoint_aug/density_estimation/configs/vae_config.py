from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "cvae"
    config.results_file = "vae_params.pkl"
    config.mode = "train"
    config.vizdom_name = "vae_toy"
    config.hidden_size = 128

    # vae
    config.cond_dim = 2
    config.latent_size = 8
    config.kl_div_weight = 0.5
    config.kl_annealing = True
    return config
