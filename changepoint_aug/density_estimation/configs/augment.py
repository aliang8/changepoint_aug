from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.exp_name = "augment"

    config.top_k = 1
    config.selection_metric = "policy_variance"
    config.selection = "per_traj"  # global
    # config.selection = "global"

    config.total_num_states = 100
    # reweight metric using density estimation
    config.reweight_with_density = False
    config.lamb = 1.3  # reweighing hyperparameter
    config.visualize = False

    # vae
    config.cond_dim = 2
    config.latent_size = 8
    config.kl_div_weight = 0.5
    config.kl_annealing = True
    config.num_posterior_samples = 100

    # policies
    config.policy_cls = "mlp"
    config.num_policies = 5

    # influence functions
    config.inf_fn_lambda = 1e-2

    config.density_exp_name = "i006_cvae"
    config.density_model_ckpt = "nt-200"
    config.density_ckpt_step = 950

    # config.exp_name = "i002_q_sarsa"
    # config.model_ckpt = "nt-25_s-0"
    config.exp_name = "i005_bc_200"
    config.model_ckpt = "nt-100_s-0"

    config.ckpt_step = 180

    # number of expert steps per state
    config.num_augmentations_per_state = 2
    config.num_expert_steps_aug = 10
    config.num_perturb_steps = 2
    config.max_states_visualize = 5

    config.augment_data_file = ""
    return config
