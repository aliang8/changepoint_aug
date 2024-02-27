"""
Train Q-function using SARSA loss

Usage:
python3 train_q_sarsa.py \
    --config=configs/q_config.py \
    --config.mode=train \
    --config.data_file=bc_policy_rollouts_100.pkl \
"""

from absl import app
from utils import load_maze_data, run_rollout_maze
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from models import q_fn
import os
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from base_trainer import BaseTrainer
import mlogger

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")


class QTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)

        # load data
        data_file = os.path.join(config.data_dir, config.data_file)
        _, self.train_dataloader, self.test_dataloader, obs_dim, action_dim = (
            load_maze_data(
                data_file,
                batch_size=config.batch_size,
                num_trajs=config.num_trajs,
                train_perc=config.train_perc,
            )
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.xp.train.q_loss = mlogger.metric.Average(
            plotter=self.plotter, plot_title="Q Loss", plot_legend="train"
        )
        self.xp.test.q_loss = mlogger.metric.Average(
            plotter=self.plotter, plot_title="Q Loss", plot_legend="test"
        )

        self.ts = self.create_ts(next(self.rng_seq))
        self.jit_update_step = jax.jit(self.update_q_step)

    def td_loss_fn(
        self, q_params, ts, obss, actions, obss_tp1, actions_tp1, rewards, dones
    ):
        # Q(s,a) -  r + gamma * (1 - done) * Q(s', pi(s'))
        q = ts.apply_fn(q_params, obss, actions)
        q = jnp.squeeze(q, axis=-1)
        q_tp1 = ts.apply_fn(q_params, obss_tp1, actions_tp1)
        q_tp1 = jnp.squeeze(q_tp1, axis=-1)
        backup = rewards + self.config.gamma * (1 - dones) * q_tp1
        loss_q = optax.squared_error(q, backup).mean()
        return loss_q

    def update_q_step(self, ts, obss, actions, obss_tp1, actions_tp1, rewards, dones):
        td_loss, grads = jax.value_and_grad(self.td_loss_fn)(
            ts.params,
            ts,
            obss,
            actions,
            obss_tp1,
            actions_tp1,
            rewards,
            dones,
        )
        ts = ts.apply_gradients(grads=grads)
        return ts, td_loss

    def create_ts(self, rng_key):
        sample_obs = jnp.zeros((1, self.obs_dim))
        sample_action = jnp.zeros((1, self.action_dim))
        rng_key, init_q_key = jax.random.split(rng_key)
        q_params = q_fn.init(
            init_q_key, sample_obs, sample_action, self.config.hidden_size
        )
        q_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(q_params))
        print(f"Number of q-function parameters: {param_count}")

        q_fn_apply = partial(
            jax.jit(hk.without_apply_rng(q_fn).apply, static_argnums=(3)),
            hidden_size=self.config.hidden_size,
        )
        train_state = TrainState.create(
            apply_fn=q_fn_apply,
            params=q_params,
            tx=q_opt,
        )
        return train_state

    def train_step(self, batch):
        obss, actions, obss_tp1, actions_tp1, rewards, dones = batch
        # update q-function
        self.ts, q_loss = self.jit_update_step(
            self.ts, obss, actions, obss_tp1, actions_tp1, rewards, dones
        )

        self.xp.train.q_loss.update(q_loss.item(), weighting=obss.shape[0])

    def test(self, epoch):
        for batch in self.test_dataloader:
            obss, actions, obss_tp1, actions_tp1, rewards, dones = batch
            q_loss = self.td_loss_fn(
                self.ts.params,
                self.ts,
                obss,
                actions,
                obss_tp1,
                actions_tp1,
                rewards,
                dones,
            )
            self.xp.test.q_loss.update(q_loss.item(), weighting=obss.shape[0])


def main(_):
    config = _CONFIG.value
    print(config)
    trainer = QTrainer(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


if __name__ == "__main__":
    app.run(main)
