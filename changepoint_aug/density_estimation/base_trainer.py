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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
import mlogger
from tensorflow_probability.substrates import jax as tfp
from utils import load_maze_data, make_blob_dataset

dist = tfp.distributions

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        # logger
        self.plotter = mlogger.VisdomPlotter(
            {
                "env": self.config.vizdom_name,
                "server": "http://localhost",
                "port": 8097,
            },
            manual_update=True,
        )

        # load dataset
        self.dataset, self.train_loader, self.test_loader = self.load_data()

        self.xp = mlogger.Container()
        self.xp.config = mlogger.Config(plotter=self.plotter)
        self.xp.config.update(**self.config)
        self.xp.train = mlogger.Container()
        self.xp.test = mlogger.Container()

    def create_ts(self, rng):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def load_data(self):
        data_file = os.path.join(self.config.data_dir, self.config.data_file)
        if self.config.dataset == "MAZE":
            dataset, train_loader, test_loader, obs_dim, action_dim = load_maze_data(
                data_file,
                batch_size=self.config.batch_size,
                num_trajs=self.config.num_trajs,
                train_perc=self.config.train_perc,
            )
        elif self.config.dataset == "TOY":
            dataset, train_loader, test_loader, obs_dim, action_dim = make_blob_dataset(
                10000,
                centers=6,
                n_features=4,
                random_state=0,
                train_perc=self.config.train_perc,
                batch_size=self.config.batch_size,
            )

        return dataset, train_loader, test_loader

    def train(self):
        with mlogger.stdout_to("printed_stuff.txt"):
            for epoch in tqdm.tqdm(range(self.config.num_epochs)):
                # reset epoch metrics
                for metric in self.xp.train.metrics():
                    metric.reset()

                for batch in self.train_loader:
                    self.train_step(batch)
                    self.global_step += 1

                if epoch % self.config.test_interval == 0:
                    # reset metrics for test
                    for metric in self.xp.test.metrics():
                        metric.reset()

                    self.test(epoch)
                    for metric in self.xp.test.metrics():
                        metric.log()

                # reset log metrics
                for metric in self.xp.train.metrics():
                    metric.log()

                self.plotter.update_plots()

                # save model
                if epoch % self.config.save_interval == 0:
                    ckpt_file = os.path.join(
                        self.config.results_dir, self.config.results_file
                    )
                    print(f"saving model at epoch {epoch} to {ckpt_file}")
                    with open(ckpt_file, "wb") as f:
                        pickle.dump(self.ts.params, f)
