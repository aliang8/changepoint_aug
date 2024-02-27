from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_base_config():
    config = ConfigDict()
    config.seed = 0
    config.data_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/datasets"
    )
    config.data_file = "bc_policy_rollouts_100.pkl"
    config.video_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/videos"
    )
    config.batch_size = 128
    config.hidden_size = 128
    config.lr = 3e-4
    config.num_epochs = 200
    config.train_perc = 0.9
    config.shuffle_dataset = True
    config.test_interval = 10
    config.save_interval = 10
    config.num_eval_episodes = 10
    config.max_episode_steps = 1000
    config.num_trajs = 200
    config.results_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/model_ckpts/"
    )
    config.mode = "train"
    config.vizdom_name = ""
    config.dataset = "TOY"
    return config
