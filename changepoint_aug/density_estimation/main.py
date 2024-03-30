from absl import app, logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import re
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig

from changepoint_aug.density_estimation.trainers import *

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


_CONFIG = config_flags.DEFINE_config_file("config")

# shorthands for config parameters
psh = {
    "batch_size": "bs",
    "env_id": "eid",
    "hidden_size": "hs",
    "max_episode_steps": "mes",
    "num_epochs": "ne",
    "train_perc": "tp",
    "trainer": "t",
    "num_trajs": "nt",
    "policy_cls": "pc",
    "num_policies": "np",
    "num_eval_episodes": "nee",
    "num_augmentation_steps": "nas",
    "seed": "s",
    "kl_div_weight": "kl",
}

# run with ray tune
param_space = {
    "seed": tune.grid_search([0, 1, 2, 3, 4]),
    "num_trajs": tune.grid_search([5, 10, 25, 50, 75]),
}


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()

    if trial_dir:
        # this is if we are running with Ray
        logging.info("trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
        # the group name is without seed
        config["group_name"] = re.sub("_s-\d", "", base_name)
        logging.info(f"wandb group name: {config['group_name']}")
    else:
        suffix = f"{config['exp_name']}_s-{config['seed']}_t-{config['trainer']}"
        config["root_dir"] = Path(config["root_dir"]) / "results" / suffix

    # wrap config in ConfigDict
    config = ConfigDict(config)

    if config.trainer == "q_sarsa":
        trainer_cls = QTrainer
    elif config.trainer == "bc":
        trainer_cls = BCTrainer
    elif config.trainer == "cvae":
        trainer_cls = CVAETrainer
    else:
        raise ValueError(f"Policy {config.policy} not implemented")

    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


def trial_str_creator(trial):
    trial_str = trial.config["exp_name"] + "_"
    for k, v in trial.config.items():
        if k in psh and k in param_space:
            trial_str += f"{psh[k]}-{v}_"
    # trial_str += str(trial.trial_id)

    trial_str = trial_str[:-1]
    logging.info(f"trial_str: {trial_str}")
    return trial_str


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        config["use_wb"] = True  # always log to wandb when we are running with ray tune
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 5, "gpu": 0.1})

        run_config = RunConfig(
            name=config["exp_name"],
            local_dir="/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/ray_results",
            storage_path="/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/ray_results",
            log_to_file=True,
        )
        tuner = tune.Tuner(
            train_model,
            param_space=config,
            run_config=run_config,
            tune_config=tune.TuneConfig(
                trial_name_creator=trial_str_creator,
                trial_dirname_creator=trial_str_creator,
            ),
        )
        results = tuner.fit()
        logging.info(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
