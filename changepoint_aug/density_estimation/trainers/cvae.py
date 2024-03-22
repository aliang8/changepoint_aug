"""
Train VAE for cond conditioned density estimation
"""

from absl import logging
import os
import io
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
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from flax.training.train_state import TrainState
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from tensorflow_probability.substrates import jax as tfp

from changepoint_aug.density_estimation.trainers.base_trainer import BaseTrainer
from changepoint_aug.density_estimation.utils import frange_cycle_linear
from changepoint_aug.density_estimation.models import vae_fn, decode_fn
from changepoint_aug.density_estimation.visualize import visualize_density_estimate

dist = tfp.distributions


def create_ts(config, obs_dim, rng_key):
    sample_obs = jnp.zeros((1, obs_dim))
    sample_cond = jnp.zeros((1, config.cond_dim))

    kwargs = dict(
        latent_size=config.latent_size,
        hidden_size=config.hidden_size,
        obs_dim=obs_dim,
        cond_dim=config.cond_dim,
    )

    if config.load_from_ckpt != "":
        vae_params = pickle.load(open(config.load_from_ckpt, "rb"))
    else:
        rng_key, init_key = jax.random.split(rng_key)
        vae_params = vae_fn.init(init_key, sample_obs, sample_cond, **kwargs)

    vae_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.lr),
    )
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    logging.info(f"Number of vae parameters: {param_count}")

    vae_fn_apply = partial(jax.jit(vae_fn.apply, static_argnums=(4, 5, 6, 7)), **kwargs)

    train_state = TrainState.create(
        apply_fn=vae_fn_apply,
        params=vae_params,
        tx=vae_opt,
    )
    return train_state


class CVAETrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        self.obs_dim = 2  # just use x,y for maze data
        # self.obs_dim = 4  # just use x,y for maze data

        self.ts = create_ts(config, self.obs_dim, next(self.rng_seq))

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
                "vae_loss": loss,
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

    def train_step(self, batch):
        if self.config.env == "MAZE":
            obs, *_ = batch

            # this is the goal location
            cond = obs[:, 4:6]
            obs = obs[:, : self.obs_dim]
        elif self.config.env == "TOY":
            obs, cond = batch
            cond = jnp.expand_dims(cond, axis=-1)

        if self.config.kl_annealing:
            kl_div_weight = self.kl_scheduler[self.global_step]
        else:
            kl_div_weight = self.config.kl_div_weight

        self.ts, vae_loss, metrics = self.jit_update_step(
            self.ts, next(self.rng_seq), obs, cond, kl_div_weight
        )
        metrics["kl_weight"] = kl_div_weight

        return metrics

    def test(self, epoch):
        avg_test_metrics = Counter()
        for batch in self.test_loader:
            if self.config.env == "MAZE":
                obs, *_ = batch
                cond = obs[:, 4:6]
                obs = obs[:, : self.obs_dim]
            elif self.config.env == "TOY":
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
            avg_test_metrics += Counter(metrics)

        for k in avg_test_metrics:
            avg_test_metrics[k] /= len(self.test_loader)

        # visualize density estimation for dataset and random trajectories
        if self.config.env == "MAZE":
            # import ipdb

            # ipdb.set_trace()
            start_indices = np.where(self.dataset[:][-1])[0]
            start_indices += 1
            start_indices = np.insert(start_indices, 0, 0)
            start_indices = start_indices[:-1]

            (all_obss, _, _, _, _, _) = self.dataset[:]
            all_goals = all_obss[start_indices, 4:6]
            # sample a couple random goals
            goals = all_goals[np.random.choice(len(all_goals), size=6, replace=False)]

            if self.config.kl_annealing:
                kl_div_weight = self.kl_scheduler[self.global_step]
            else:
                kl_div_weight = self.config.kl_div_weight

            fig = visualize_density_estimate(
                self.ts,
                next(self.rng_seq),
                self.config,
                all_obss.numpy(),
                goals.numpy(),
                kl_div_weight,
            )

            if self.wandb_run:
                # save as image instead of plotly interactive figure
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                wandb.log(({"viz/density_estimation": wandb.Image(Image.open(buf))}))

        return avg_test_metrics
