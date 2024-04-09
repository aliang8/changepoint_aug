from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.trainer = "cvae"
    config.mode = "train"
    config.hidden_size = 128
    config.save_key = "test/vae_loss"
    config.best_metric = "min"

    config.num_epochs = 1000
    config.test_interval = 50
    config.save_interval = 50

    # vae
    config.cond_dim = 2
    config.latent_size = 8
    config.kl_div_weight = 0.5
    config.kl_annealing = True

    # for visualization
    config.num_posterior_samples = 100
    return config
