from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_base_config():
    config = ConfigDict()
    config.seed = 0
    config.exp_name = "i001"
    config.root_dir = "/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation"
    config.data_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/datasets"
    )
    config.data_file = "sac_maze_dense_reward_200_episodes.pkl"
    config.video_dir = "videos"
    config.batch_size = 128
    config.hidden_size = 128
    config.lr = 3e-4
    config.num_epochs = 200
    config.train_perc = 0.9
    config.shuffle_dataset = True
    config.test_interval = 10
    config.save_interval = 20
    config.num_eval_episodes = 10
    config.max_episode_steps = 1000
    config.num_trajs = 500
    config.ckpt_dir = "model_ckpts"
    config.mode = "train"
    config.vizdom_name = ""
    config.env = "MAZE"
    config.env_id = "button-press-v2"  # metaworld specific
    config.disable_tqdm = False
    config.save_video = False
    config.smoke_test = False
    config.logger_cls = "wandb"
    config.use_wb = False
    config.notes = ""
    config.tags = []

    return config
