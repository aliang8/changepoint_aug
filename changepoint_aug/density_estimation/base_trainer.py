"""
Base Trainer

Usage:
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
import wandb
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
import mlogger
from pathlib import Path
from tensorflow_probability.substrates import jax as tfp
from utils import load_pkl_dataset, make_blob_dataset

dist = tfp.distributions


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        if self.config.logger_cls == "visdom":
            # logger
            self.plotter = mlogger.VisdomPlotter(
                {
                    "env": self.config.vizdom_name,
                    "server": "http://localhost",
                    "port": 8097,
                },
                manual_update=True,
            )
            self.wandb_run = None
            self.xp = mlogger.Container()
            self.xp.config = mlogger.Config(plotter=self.plotter)
            self.xp.config.update(**self.config)
            self.xp.train = mlogger.Container()
            self.xp.test = mlogger.Container()
        elif self.config.logger_cls == "wandb" and self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="data_augmentation",
                name="test",
                notes=self.config.notes,
                tags=self.config.tags,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # load dataset
        self.dataset, self.train_loader, self.test_loader = self.load_data()

        # setup log dirs
        self.ckpt_dir = Path(self.config.root_dir) / self.config.ckpt_dir
        print("ckpt_dir: ", self.ckpt_dir)

        # make it
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = Path(self.config.root_dir) / self.config.video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        if self.config.logger_cls == "vizdom":
            print("loss keys: ", self.loss_keys)
            for lk, logger_cls, mode, title in self.loss_keys:
                if mode == "both":
                    self.xp.train.__setattr__(
                        lk,
                        logger_cls(
                            plotter=self.plotter, plot_title=title, plot_legend="train"
                        ),
                    )
                    self.xp.test.__setattr__(
                        lk,
                        logger_cls(
                            plotter=self.plotter, plot_title=title, plot_legend="test"
                        ),
                    )
                else:
                    self.xp.__getattribute__(mode).__setattr__(
                        lk,
                        logger_cls(
                            plotter=self.plotter, plot_title=title, plot_legend=mode
                        ),
                    )

    def create_ts(self, rng):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def load_data(self):
        data_file = Path(self.config.data_dir) / self.config.data_file
        if self.config.env == "MAZE" or self.config.env == "MW":
            dataset, train_loader, test_loader, obs_dim, action_dim = load_pkl_dataset(
                data_file,
                batch_size=self.config.batch_size,
                num_trajs=self.config.num_trajs,
                train_perc=self.config.train_perc,
                env=self.config.env,
            )
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            print("obs_dim: ", obs_dim, " action_dim: ", action_dim)
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
        # with mlogger.stdout_to("printed_stuff.txt"):
        for epoch in tqdm.tqdm(
            range(self.config.num_epochs), disable=self.config.disable_tqdm
        ):
            if self.config.logger_cls == "vizdom":
                # reset epoch metrics
                for metric in self.xp.train.metrics():
                    metric.reset()

            for batch in self.train_loader:
                train_metrics = self.train_step(batch)
                self.global_step += 1

                if self.config.logger_cls == "vizdom":
                    for lk in train_metrics.keys():
                        self.xp.train.__getattribute__(lk).update(
                            train_metrics[lk].item(), weighting=batch[0].shape[0]
                        )
                elif self.wandb_run:
                    # add prefix to metrics
                    train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                    self.wandb_run.log(train_metrics, step=self.global_step)

            if epoch % self.config.test_interval == 0:
                if self.config.logger_cls == "vizdom":
                    # reset metrics for test
                    for metric in self.xp.test.metrics():
                        metric.reset()

                test_metrics = self.test(epoch)

                if self.config.logger_cls == "vizdom":
                    for metric in self.xp.test.metrics():
                        metric.log()
                elif self.wandb_run:
                    test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
                    self.wandb_run.log(test_metrics, step=self.global_step)

            if self.config.logger_cls == "vizdom":
                # reset log metrics
                for metric in self.xp.train.metrics():
                    metric.log()

                self.plotter.update_plots()

            # save model
            if epoch % self.config.save_interval == 0:
                ckpt_file = self.ckpt_dir / f"{self.config.env}_epoch_{epoch}.pkl"
                print(f"saving model at epoch {epoch} to {ckpt_file}")
                with open(ckpt_file, "wb") as f:
                    pickle.dump(self.ts.params, f)
