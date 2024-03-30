"""
Base Trainer

Usage:
"""

from absl import app, logging
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
import torch
import wandb
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from pathlib import Path
from tensorflow_probability.substrates import jax as tfp

from changepoint_aug.density_estimation.data import load_pkl_dataset, make_blob_dataset

dist = tfp.distributions


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        # set torch seed to maintain reproducibility
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="data_augmentation",
                name=config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                group=config.group_name if config.group_name else None,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # load dataset
        self.dataset, self.train_loader, self.test_loader = self.load_data()

        self.root_dir = Path(self.config.root_dir)

        # setup log dirs
        self.ckpt_dir = self.root_dir / self.config.ckpt_dir
        logging.info(f"ckpt_dir: {self.ckpt_dir}")

        # make it
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.root_dir / self.config.video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if config.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def load_data(self):
        if self.config.env == "MAZE" or self.config.env == "MW":
            dataset, train_loader, test_loader, obs_dim, action_dim, _ = (
                load_pkl_dataset(
                    self.config.data_dir,
                    self.config.data_file,
                    batch_size=self.config.batch_size,
                    num_trajs=self.config.num_trajs,
                    train_perc=self.config.train_perc,
                    env=self.config.env,
                    augmentation_data_files=self.config.augmentation_data_files,
                    num_augmentation_steps=self.config.num_augmentation_steps,
                )
            )
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            logging.info(f"obs_dim: {obs_dim} action_dim: {action_dim}")

        elif self.config.env == "TOY":
            dataset, train_loader, test_loader, obs_dim, action_dim = make_blob_dataset(
                10000,
                # centers=3,
                n_features=2,
                random_state=0,
                train_perc=self.config.train_perc,
                batch_size=self.config.batch_size,
                centers=[[0, 2], [-1, -1], [1, -1]],
            )

        return dataset, train_loader, test_loader

    def train(self):
        # run one eval before training
        test_metrics = self.test(0)

        for epoch in tqdm.tqdm(
            range(1, self.config.num_epochs + 1), disable=self.config.disable_tqdm
        ):
            # run a test first
            if epoch % self.config.test_interval == 0:
                test_time = time.time()
                test_metrics = self.test(epoch)
                test_total_time = time.time() - test_time

                # save based on key
                if self.config.save_key in test_metrics:
                    key = self.config.save_key
                    if (
                        self.config.best_metric == "max"
                        and test_metrics[key] > self.best_metric
                    ) or (
                        self.config.best_metric == "min"
                        and test_metrics[key] < self.best_metric
                    ):
                        self.best_metric = test_metrics[key]
                        ckpt_file = self.ckpt_dir / f"best.pkl"
                        logging.info(
                            f"new best value: {test_metrics[key]}, saving best model at epoch {epoch} to {ckpt_file}"
                        )
                        with open(ckpt_file, "wb") as f:
                            pickle.dump(self.ts.params, f)

                        # create a file with the best metric in the name, use a placeholder
                        best_ckpt_file = self.ckpt_dir / "best.txt"
                        with open(best_ckpt_file, "w") as f:
                            f.write(f"{epoch}, {test_metrics[key]}")

                test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
                if self.wandb_run is not None:
                    self.wandb_run.log(test_metrics)
                    self.wandb_run.log({"time/test_time": test_total_time})

            epoch_time = time.time()
            for batch in self.train_loader:
                batch_time = time.time()
                train_metrics = self.train_step(batch)
                batch_total_time = time.time() - batch_time

                self.global_step += 1

                if self.wandb_run is not None:
                    # add prefix to metrics
                    train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                    self.wandb_run.log(train_metrics)
                    self.wandb_run.log({"time/batch_time": batch_total_time})

            epoch_total_time = time.time() - epoch_time

            # log time information
            if self.wandb_run is not None:
                self.wandb_run.log({"time/epoch_time": epoch_total_time})

            # save model at set intervals
            if epoch % self.config.save_interval == 0:
                ckpt_file = self.ckpt_dir / f"epoch_{epoch}.pkl"
                logging.info(f"saving model at epoch {epoch} to {ckpt_file}")
                with open(ckpt_file, "wb") as f:
                    pickle.dump(self.ts.params, f)
