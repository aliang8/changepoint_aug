from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.exp_name = "augment"
    config.load_randomly_sampled_states = False

    config.top_k = 1
    config.selection_metric = "policy_variance"
    config.selection = "per_traj"  # global
    # config.selection = "global"

    config.total_num_states = 100
    # reweight metric using density estimation
    config.reweight_with_density = False
    config.lamb = 1.3  # reweighing hyperparameter
    config.visualize = False
    config.metric_threshold = (
        10  # threshold for metric, only sample states with metric > threshold
    )

    # number of augmentations for each sampled state
    config.num_augmentations_per_state = 1
    # number of expert steps per state
    config.num_expert_steps_aug = 10
    # number of random steps to take to perturb state
    config.num_perturb_steps = 1
    config.max_states_visualize = 5

    # for q-variance metric
    config.num_samples_qvar = 100

    # vae
    config.cond_dim = 2
    config.latent_size = 8
    config.kl_div_weight = 0.5
    config.kl_annealing = True
    config.num_posterior_samples = 100

    # policies
    config.policy_cls = "gaussian"
    config.num_policies = 5

    # influence functions
    config.inf_fn_lambda = 1e-2

    # for loading model to compute the density
    config.density_exp_name = "i006_cvae"
    config.density_model_ckpt = "nt-200"
    config.density_ckpt_step = 950

    # for loading model to compute the heuristic
    config.exp_name = "i032_bc"
    config.model_ckpt = (
        "i032_bc_s-0_nt-5_eid-maze_2d_wall_v0_nat-0_nas-0_ne-5000_np-5_pc-gaussian_t-bc"
    )
    config.ckpt_step = 5000

    config.oracle_model_ckpt_path = "/scr/aliang80/active_imitation_learning/active_imitation_learning/model_ckpts/sac_maze_5M.pt"

    config.keys_to_include = {
        "env": None,
        "num_expert_steps_aug": None,
        "num_augmentations_per_state": None,
        "num_perturb_steps": None,
        "selection": None,
        "selection_metric": None,
        "reweight_with_density": None,
        # "total_num_states": None,
    }
    return config
