from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.results_file = "vae_params.pkl"
    config.mode = "train"
    config.vizdom_name = "vae_toy"
    config.hidden_size = 64

    # vae
    config.cond_dim = 2
    config.latent_size = 1
    config.kl_div_weight = 1e-1
    config.kl_annealing = True
    return config
