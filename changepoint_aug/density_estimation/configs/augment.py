from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.exp_name = "augment"

    config.top_k = 2
    config.selection_metric = "policy_variance"
    # reweight metric using density estimation
    config.reweight_with_density = False

    # vae
    config.cond_dim = 2
    config.latent_size = 8
    config.kl_div_weight = 0.5
    config.kl_annealing = True
    config.num_posterior_samples = 100

    # policies
    config.policy_cls = "mlp"
    config.num_policies = 5

    config.density_exp_name = "i006_cvae"
    config.density_model_ckpt = "nt-200"
    config.density_ckpt_step = 950

    # config.exp_name = "i002_q_sarsa"
    # config.model_ckpt = "nt-25_s-0"
    config.exp_name = "i005_bc_200"
    config.model_ckpt = "nt-100_s-0"

    config.ckpt_step = 180

    # number of expert steps per state
    config.num_expert_steps_aug = 10
    config.num_perturb_steps = 0
    config.max_states_visualize = 10

    config.augment_data_file = f"augment_dataset_aug-{config.num_expert_steps_aug}_m-{config.selection_metric}.pkl"
    return config
