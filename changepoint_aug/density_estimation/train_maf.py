"""
Conditional density estimation with MADE and MAF
MADE: https://arxiv.org/abs/1502.03509

p(obs | goal)
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
from base_trainer import BaseTrainer
from utils import (
    NumpyLoader,
    create_learning_rate_fn,
    load_maze_data,
    frange_cycle_linear,
)
from models import maf_fn
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions

_CONFIG = config_flags.DEFINE_config_file("config")

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class MAFTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)

        # load data
        self.ts = self.create_ts(next(self.rng_seq))

        self.obs_dim = self.dataset[:][0].shape[-1] - 2
        self.goal_dim = 2

        def maf_loss_fn(params, ts, rng_key, obs, goal):
            loss = -model.log_prob(x, y if args.cond_label_size else None).mean(0)
            return

        def update_step(ts, rng_key, obs, goal, kl_div_weight):
            (maf_loss, metrics), grads = jax.value_and_grad(maf_loss_fn, has_aux=True)(
                ts.params, ts, rng_key, obs, goal
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, maf_loss, metrics

        self.jit_update_step = jax.jit(update_step)

    def create_ts(self, rng_key):
        rng_key, init_key = jax.random.split(rng_key)
        sample_obs = jnp.zeros((1, self.obs_dim))
        sample_goal = jnp.zeros((1, self.goal_dim))

        kwargs = dict(
            hiden_size=self.config.hidden_size,
            num_layers=1,
            input_size=self.obs_dim,
            cond_size=self.goal_dim,
        )
        maf_params = maf_fn.init(init_key, sample_obs, sample_goal, **kwargs)

        maf_fn_apply = partial(jax.jit(maf_fn.apply), **kwargs)
        maf_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        train_state = TrainState.create(
            apply_fn=maf_fn_apply,
            params=maf_params,
            tx=maf_opt,
        )
        return train_state

    def train_step(self, batch):
        obs, *_ = batch
        obs = obs[:, : self.obs_dim]
        goal = obs[:, -2:]

        self.ts, loss, metrics = self.jit_update_step(
            self.ts, next(self.rng_seq), obs, goal
        )

    def test(self, epoch):
        pass


def main(_):
    config = _CONFIG.value
    print(config)
    trainer = MAFTrainer(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


if __name__ == "__main__":
    app.run(main)
