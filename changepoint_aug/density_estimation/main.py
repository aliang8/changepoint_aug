"""
Train Q-function 

Usage:
python3 main.py \
    --config=configs/q_sarsa_config.py \
    --config.mode=train \
    
python3 main.py \
    --config=configs/bc_config.py \
    --config.mode=train \
    --config.exp_name=2_bc

python3 main.py \
    --config=configs/vae_config.py \
    --config.mode=train \
    --config.exp_name=2_vae
"""

from absl import app
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import mlogger
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from train_q_sarsa import QTrainer
from train_bc import BCTrainer
from train_gc_vae_de import CVAETrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
}

# run with ray tune
param_space = {
    # "latent_dim": tune.grid_search([5, 8]),
    # "kl_div_weight": tune.grid_search([1e-1, 1e-2]),
    # "seed": tune.grid_search([0, 1]),
    # "lr": tune.grid_search([1e-3, 1e-4]),
    # "hidden_size": tune.grid_search([64, 128, 256]),
    # "gamma": tune.grid_search([0.99, 0.9]),
    "num_trajs": tune.grid_search([5]),
}


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
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
    trial_str = ""
    for k, v in trial.config.items():
        if k in psh and k in param_space:
            trial_str += f"{psh[k]}-{v}_"
    # trial_str += str(trial.trial_id)
    print("trial_str: ", trial_str)
    return trial_str


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 1, "gpu": 0.2})

        run_config = RunConfig(
            name=config["exp_name"],
            local_dir="/scr/aliang80/changepoint_aug/changepoint_aug/ray_results",
            storage_path="/scr/aliang80/changepoint_aug/changepoint_aug/ray_results",
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
        print(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
