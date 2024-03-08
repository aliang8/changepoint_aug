"""
Train VAE for cond conditioned density estimation
"""

import os
import jax
import haiku as hk
import numpy as np
import optax
import jax.numpy as jnp
import dataclasses
from typing import NamedTuple
import pickle
import tqdm
import torch
from utils import frange_cycle_linear
from models import vae_fn
from flax.training.train_state import TrainState
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from base_trainer import BaseTrainer
from functools import partial
from absl import app
import mlogger
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


class CVAETrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        self.loss_keys = [
            ("kl_div_weight", mlogger.metric.Simple, "both", "KL Weight"),
            ("kl_div_loss", mlogger.metric.Average, "both", "KL Div Loss"),
            ("recon_loss", mlogger.metric.Average, "both", "Reconstruction Loss"),
            (
                "prior_matching_loss",
                mlogger.metric.Average,
                "both",
                "Prior Matching Loss",
            ),
        ]

        super().__init__(config)
        # self.obs_dim = self.dataset[:][0].shape[-1] - 2
        self.obs_dim = 2  # just use x,y for maze data
        self.cond_dim = self.config.cond_dim

        self.ts = self.create_ts(next(self.rng_seq))
        self.kl_scheduler = frange_cycle_linear(
            self.config.num_epochs * len(self.train_loader),
            start=0.0,
            stop=self.config.kl_div_weight,
            n_cycle=4,
        )

        def vae_loss_fn(vae_params, ts, rng_key, obs, cond, kl_div_weight):
            vae_sample_key, _ = jax.random.split(rng_key)
            vae_output = ts.apply_fn(vae_params, vae_sample_key, obs, cond)
            obs_pred = vae_output.recon
            recon_loss = optax.squared_error(obs_pred, obs)
            # sum over obs dimension
            recon_loss = recon_loss.sum(axis=-1).mean()

            # regularize to uniform prior
            prior = dist.Normal(0, 1)
            posterior = dist.Normal(loc=vae_output.mean, scale=vae_output.stddev)
            kl_div_loss = dist.kl_divergence(posterior, prior)
            # jax.debug.breakpoint()

            # sum over latent dim
            kl_div_loss = kl_div_loss.sum(axis=-1).mean()

            # train conditional prior matching
            prior_dist = dist.Normal(vae_output.prior_mean, vae_output.prior_stddev)
            posterior_dist = dist.Normal(vae_output.mean, vae_output.stddev)
            # only train the prior
            prior_matching_loss = dist.kl_divergence(
                jax.lax.stop_gradient(posterior_dist), prior_dist
            )
            prior_matching_loss = prior_matching_loss.sum(axis=-1).mean()

            loss = recon_loss + kl_div_weight * kl_div_loss + prior_matching_loss
            metrics = {
                "recon_loss": recon_loss,
                "kl_div_loss": kl_div_loss,
                "prior_matching_loss": prior_matching_loss,
            }
            return loss, metrics

        def update_step(ts, rng_key, obs, cond, kl_div_weight):
            (vae_loss, metrics), grads = jax.value_and_grad(vae_loss_fn, has_aux=True)(
                ts.params, ts, rng_key, obs, cond, kl_div_weight
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, vae_loss, metrics

        self.jit_update_step = jax.jit(update_step)
        self.jit_loss_fn = jax.jit(vae_loss_fn)

    def create_ts(self, rng_key):
        sample_obs = jnp.zeros((1, self.obs_dim))
        sample_cond = jnp.zeros((1, self.cond_dim))
        rng_key, init_key = jax.random.split(rng_key)

        kwargs = dict(
            latent_size=self.config.latent_size,
            hidden_size=self.config.hidden_size,
            obs_dim=self.obs_dim,
            cond_dim=self.cond_dim,
        )

        vae_params = vae_fn.init(init_key, sample_obs, sample_cond, **kwargs)
        vae_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        print(f"Number of vae parameters: {param_count}")

        vae_fn_apply = partial(
            jax.jit(vae_fn.apply, static_argnums=(4, 5, 6, 7)), **kwargs
        )

        train_state = TrainState.create(
            apply_fn=vae_fn_apply,
            params=vae_params,
            tx=vae_opt,
        )
        return train_state

    def train_step(self, batch):
        if self.config.dataset == "MAZE":
            obs, *_ = batch
            cond = obs[:, -self.cond_dim :]
            obs = obs[:, : self.obs_dim]
        elif self.config.dataset == "TOY":
            obs, cond = batch
            cond = jnp.expand_dims(cond, axis=-1)

        if self.config.kl_annealing:
            kl_div_weight = self.kl_scheduler[self.global_step]
        else:
            kl_div_weight = self.config.kl_div_weight

        self.xp.train.kl_div_weight.update(kl_div_weight)

        self.ts, vae_loss, metrics = self.jit_update_step(
            self.ts, next(self.rng_seq), obs, cond, kl_div_weight
        )
        return metrics

    def test(self, epoch):
        for batch in self.test_loader:
            if self.config.dataset == "MAZE":
                obs, *_ = batch
                cond = obs[:, -2:]
                obs = obs[:, : self.obs_dim]
            elif self.config.dataset == "TOY":
                obs, cond = batch
                cond = jnp.expand_dims(cond, axis=-1)

            loss, metrics = self.jit_loss_fn(
                self.ts.params,
                self.ts,
                next(self.rng_seq),
                obs,
                cond,
                self.config.kl_div_weight,
            )

            for lk in metrics.keys():
                self.xp.test.__getattribute__(lk).update(
                    metrics[lk].item(), weighting=obs.shape[0]
                )
