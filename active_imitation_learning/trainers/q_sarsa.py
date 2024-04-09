"""
Train Q-function using SARSA loss
"""

from absl import logging
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
import io
import wandb
from PIL import Image
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
import utils as utils
import matplotlib.pyplot as plt
from collections import Counter

from active_imitation_learning.trainers.base_trainer import BaseTrainer


def create_ts(config, obs_dim, action_dim, rng_key):
    sample_obs = jnp.zeros((1, obs_dim))
    sample_action = jnp.zeros((1, action_dim))

    if config.load_from_ckpt != "":
        q_params = pickle.load(open(config.load_from_ckpt, "rb"))
    else:
        rng_key, init_q_key = jax.random.split(rng_key)
        q_params = q_fn.init(init_q_key, sample_obs, sample_action, config.hidden_size)

    q_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.lr),
    )
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(q_params))
    logging.info(f"Number of q-function parameters: {param_count}")

    q_fn_apply = partial(
        jax.jit(hk.without_apply_rng(q_fn).apply, static_argnums=(3)),
        hidden_size=config.hidden_size,
    )
    train_state = TrainState.create(
        apply_fn=q_fn_apply,
        params=q_params,
        tx=q_opt,
    )
    return train_state


class QTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        self.ts = create_ts(config, self.obs_dim, self.action_dim, next(self.rng_seq))
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

        metrics = {"q_loss": loss_q}
        return loss_q, metrics

    def update_q_step(self, ts, obss, actions, obss_tp1, actions_tp1, rewards, dones):
        (td_loss, metrics), grads = jax.value_and_grad(self.td_loss_fn, has_aux=True)(
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
        return ts, td_loss, metrics

    def train_step(self, batch):
        obss, actions, obss_tp1, actions_tp1, rewards, dones = batch

        obss = obss[:, : self.obs_dim]
        obss_tp1 = obss_tp1[:, : self.obs_dim]

        # update q-function
        self.ts, q_loss, metrics = self.jit_update_step(
            self.ts, obss, actions, obss_tp1, actions_tp1, rewards, dones
        )
        return metrics

    def test(self, epoch):
        avg_test_metrics = Counter()
        for batch in self.test_loader:
            obss, actions, obss_tp1, actions_tp1, rewards, dones = batch

            obss = obss[:, : self.obs_dim]
            obss_tp1 = obss_tp1[:, : self.obs_dim]

            q_loss, metrics = self.td_loss_fn(
                self.ts.params,
                self.ts,
                obss,
                actions,
                obss_tp1,
                actions_tp1,
                rewards,
                dones,
            )
            avg_test_metrics += Counter(metrics)

        for k in avg_test_metrics:
            avg_test_metrics[k] /= len(self.test_loader)

        if self.config.env == "MAZE":
            # visualize Q-values for random trajectory
            start_indices = np.where(self.dataset[:][-1])[0]
            start_indices += 1
            start_indices = np.insert(start_indices, 0, 0)
            start_indices = start_indices[:-1]
            (all_obss, all_actions, _, _, all_rewards, _) = self.dataset[:]

            # select random trajectory
            traj_indx = np.random.randint(0, len(start_indices) - 1)
            start = start_indices[traj_indx]
            end = start_indices[traj_indx + 1]
            obss = all_obss[start:end].numpy()

            # [x,y,x_vel,y_vel]
            goal = obss[0][4:6]
            obss = obss[:, : self.obs_dim]
            actions = all_actions[start:end].numpy()
            rewards = all_rewards[start:end].numpy()
            fig = utils.visualize_q_trajectory(
                self.ts,
                self.config.env,
                next(self.rng_seq),
                obss,
                actions,
                rewards,
                goal,
            )

            if self.wandb_run is not None:
                # save as image instead of plotly interactive figure
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=300)
                buf.seek(0)
                wandb.log(({"viz/q_trajectory": wandb.Image(Image.open(buf))}))

        return avg_test_metrics
