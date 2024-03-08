"""
Train Q-function and Policy
"""

from absl import app
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from models import QFunction, Policy, q_fn, policy_fn
import os
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any, Callable
from flax import struct
import mlogger
from base_trainer import BaseTrainer


class QTrainState(TrainState):
    apply_fn: Callable = struct.field(pytree_node=False)
    apply_actor_fn: Callable = struct.field(pytree_node=False)
    tx_actor: optax.GradientTransformation
    q_params: flax.core.frozen_dict.FrozenDict
    target_q_params: flax.core.frozen_dict.FrozenDict
    q_opt_state: flax.core.frozen_dict.FrozenDict
    actor_params: flax.core.frozen_dict.FrozenDict
    actor_opt_state: flax.core.frozen_dict.FrozenDict


class QTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        self.loss_keys = [
            ("q_loss", mlogger.metric.Average, "Q Loss"),
            ("actor_loss", mlogger.metric.Average, "Actor Loss"),
        ]
        super().__init__(config)
        self.ts = self.create_ts(next(self.rng_seq))
        self.jit_update_q_step = jax.jit(self.update_q_step)
        self.jit_update_actor_step = jax.jit(self.update_actor_step)

    def td_loss_fn(
        self,
        q_params,
        ts,
        target_q_params,
        actor_params,
        obss,
        actions,
        obss_tp1,
        rewards,
        dones,
    ):
        # Q(s,a) -  (r + gamma * (1 - done) * Q(s', pi(s')))
        q = ts.apply_fn(q_params, obss, actions)
        q = jnp.squeeze(q, axis=-1)
        action_tp1 = ts.apply_actor_fn(actor_params, obss_tp1)
        q_tp1 = ts.apply_fn(target_q_params, obss_tp1, action_tp1)
        q_tp1 = jnp.squeeze(q_tp1, axis=-1)
        backup = rewards + self.config.gamma * (1 - dones) * q_tp1
        loss_q = optax.squared_error(q, backup).mean()
        metrics = {"q_loss": loss_q}
        return loss_q, metrics

    def actor_loss_fn(self, actor_params, ts, q_params, obs):
        # maximize q value
        # E[Q(s, pi(s))]
        policy_action = ts.apply_actor_fn(actor_params, obs)
        q = ts.apply_fn(q_params, obs, policy_action)
        q = jnp.squeeze(q, axis=-1)
        # maybe add entropy
        actor_loss = -q.mean()
        metrics = {"actor_loss": actor_loss}
        return actor_loss, metrics

    def update_q_step(self, ts, obss, actions, obss_tp1, rewards, dones):
        (td_loss, metrics), grads = jax.value_and_grad(self.td_loss_fn, has_aux=True)(
            ts.q_params,
            ts,
            ts.target_q_params,
            ts.actor_params,
            obss,
            actions,
            obss_tp1,
            rewards,
            dones,
        )
        updates, q_opt_state = ts.tx.update(grads, ts.q_opt_state)
        new_params = optax.apply_updates(ts.q_params, updates)
        return new_params, q_opt_state, td_loss, metrics

    def update_actor_step(self, ts, obss):
        (actor_loss, metrics), grads = jax.value_and_grad(
            self.actor_loss_fn, has_aux=True
        )(ts.actor_params, ts, ts.q_params, obss)
        updates, actor_opt_state = ts.tx_actor.update(grads, ts.actor_opt_state)
        new_params = optax.apply_updates(ts.actor_params, updates)
        return new_params, actor_opt_state, actor_loss, metrics

    def create_ts(self, rng_key):
        sample_obs = jnp.zeros((1, self.obs_dim))
        sample_action = jnp.zeros((1, self.action_dim))
        rng_key, init_q_key = jax.random.split(rng_key)
        q_params = q_fn.init(
            init_q_key, sample_obs, sample_action, self.config.hidden_size
        )
        target_q_params = q_params.copy()
        q_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        q_opt_state = q_opt.init(q_params)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(q_params))
        print(f"Number of q-function parameters: {param_count}")

        _, init_actor_key = jax.random.split(rng_key)
        actor_params = policy_fn.init(
            init_actor_key, sample_obs, self.config.hidden_size, self.action_dim
        )
        actor_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        actor_opt_state = actor_opt.init(actor_params)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(actor_params))
        print(f"Number of policy parameters: {param_count}")

        # create
        q_fn_apply = partial(
            jax.jit(hk.without_apply_rng(q_fn).apply, static_argnums=(3)),
            hidden_size=self.config.hidden_size,
        )
        actor_fn_apply = partial(
            jax.jit(hk.without_apply_rng(policy_fn).apply, static_argnums=(2, 3)),
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
        )

        train_state = QTrainState.create(
            apply_fn=q_fn_apply,
            apply_actor_fn=actor_fn_apply,
            params=q_params,
            tx=q_opt,
            tx_actor=actor_opt,
            q_params=q_params,
            target_q_params=target_q_params,
            q_opt_state=q_opt_state,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
        )
        return train_state

    def train_step(self, batch):
        obss, actions, obss_tp1, _, rewards, dones = batch

        # update q-function
        q_params, q_opt_state, batch_q_loss, metrics = self.jit_update_q_step(
            self.ts, obss, actions, obss_tp1, rewards, dones
        )

        # update target params
        if self.global_step % self.config.target_update_freq == 0:
            target_q_params = jax.tree_map(
                lambda p, tp: tp * self.config.tau + p * (1 - self.config.tau),
                q_params,
                self.ts.target_q_params,
            )
        else:
            target_q_params = self.ts.target_q_params

        self.ts = self.ts.replace(
            q_params=q_params, target_q_params=target_q_params, q_opt_state=q_opt_state
        )
        # update actor
        actor_params, actor_opt_state, batch_actor_loss, actor_metrics = (
            self.jit_update_actor_step(self.ts, obss)
        )
        metrics.update(actor_metrics)
        self.ts = self.ts.replace(
            actor_params=actor_params, actor_opt_state=actor_opt_state
        )
        return metrics

    def test(self, epoch):
        for batch in self.test_loader:
            obss, actions, obss_tp1, _, rewards, dones = batch
            batch_q_loss, metrics = self.td_loss_fn(
                self.ts.q_params,
                self.ts.target_q_params,
                self.ts.actor_params,
                obss,
                actions,
                obss_tp1,
                rewards,
                dones,
            )
            batch_actor_loss, actor_metrics = self.actor_loss_fn(
                self.ts.actor_params, self.ts.q_params, obss
            )

            metrics.update(actor_metrics)

            for lk in metrics.keys():
                self.xp.test.__getattribute__(lk).update(
                    metrics[lk].item(), weighting=obss.shape[0]
                )
