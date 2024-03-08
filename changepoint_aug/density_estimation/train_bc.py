from absl import app
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from models import GaussianPolicy
import os
import tqdm
import pickle
import time
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
from functools import partial
from ml_collections import config_flags
from flax.training.train_state import TrainState
import mlogger
from models import policy_fn, gaussian_policy_fn
from base_trainer import BaseTrainer
from tensorflow_probability.substrates import jax as tfp
import utils as utils
import matplotlib.pyplot as plt

dist = tfp.distributions


class BCTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        self.loss_keys = [
            ("bc_loss", mlogger.metric.Average, "both", "BC Loss"),
            ("average_return", mlogger.metric.Average, "test", "Eval Returns"),
            (
                "average_success_rate",
                mlogger.metric.Average,
                "test",
                "Eval Success Rate",
            ),
            ("average_length", mlogger.metric.Average, "test", "Eval Episode Length"),
        ]
        super().__init__(config)
        self.ts = self.create_ts(next(self.rng_seq))
        self.jit_update_step = jax.jit(self.update_step)

    def bc_loss_fn(self, params, ts, obss, actions, rng_key):
        policy_rng_keys = jax.random.split(rng_key, self.config.num_policies + 1)

        if self.config.policy_cls == "mlp":
            action_pred = jax.vmap(
                lambda param, rng_key: ts.apply_fn(param, rng_key, obss)
            )(params, policy_rng_keys[1:])
            # repeat actions for each policy
            actions = jnp.repeat(actions[None], self.config.num_policies, axis=0)
            bc_loss = optax.squared_error(action_pred, actions).mean()
        elif self.config.policy_cls == "gaussian":
            _, _, mean, stddev = jax.vmap(
                lambda param, rng_key: ts.apply_fn(param, rng_key, obss)
            )(params, policy_rng_keys[1:])

            action_dist = dist.Normal(mean, stddev)
            logp = action_dist.log_prob(actions)
            logp = logp.sum(axis=-1).mean()
            bc_loss = -logp
        metrics = {"bc_loss": bc_loss}
        return bc_loss, metrics

    def update_step(self, ts, rng_key, obss, actions):
        (bc_loss, metrics), grads = jax.value_and_grad(self.bc_loss_fn, has_aux=True)(
            ts.params, ts, obss, actions, rng_key
        )
        ts = ts.apply_gradients(grads=grads)
        return ts, bc_loss, metrics

    def create_ts(self, rng_key):
        sample_obs = jnp.zeros((1, self.obs_dim))
        policy_rng_keys = jax.random.split(rng_key, self.config.num_policies + 1)

        policy_apply = (
            policy_fn if self.config.policy_cls == "mlp" else gaussian_policy_fn
        )

        params = jax.vmap(policy_apply.init, in_axes=(0, None, None, None))(
            policy_rng_keys[1:], sample_obs, self.config.hidden_size, self.action_dim
        )
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
        )
        # create
        policy_apply = partial(
            jax.jit(policy_apply.apply, static_argnums=(3, 4)),
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
        )

        ts = TrainState.create(
            apply_fn=policy_apply,
            params=params,
            tx=opt,
        )

        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {param_count}")
        return ts

    def train_step(self, batch):
        obss, actions, *_ = batch
        obss = obss[:, : self.obs_dim]

        # update model
        self.ts, _, metrics = self.jit_update_step(
            self.ts, next(self.rng_seq), obss, actions
        )
        return metrics

    def test(self, epoch):
        for batch in self.test_loader:
            obss, actions, *_ = batch

            obss = obss[:, : self.obs_dim]

            bc_loss, metrics = self.bc_loss_fn(
                self.ts.params, self.ts, obss, actions, next(self.rng_seq)
            )

            if self.config.logger_cls == "vizdom":
                for lk in metrics.keys():
                    self.xp.test.__getattribute__(lk).update(
                        metrics[lk].item(), weighting=obss.shape[0]
                    )

        # run rollouts to test trained policy
        rollouts, metrics = utils.run_rollouts(
            self.ts,
            rng_key=next(self.rng_seq),
            config=self.config,
        )

        if self.config.logger_cls == "vizdom":
            for lk in metrics.keys():
                if isinstance(metrics[lk], (int, float)):
                    metric = metrics[lk]
                else:
                    metric = metrics[lk].item()
                self.xp.test.__getattribute__(lk).update(metric, weighting=1)

        # visualize
        # visualize variance over policy ensemble for random trajectory
        start_indices = np.where(self.dataset[:][-1])[0]
        start_indices += 1
        start_indices = np.insert(start_indices, 0, 0)
        (all_obss, _, _, _, _, _) = self.dataset[:]

        # select random trajectory
        traj_indx = np.random.randint(0, len(start_indices) - 1)
        start = start_indices[traj_indx]
        end = start_indices[traj_indx + 1]
        obss = all_obss[start:end].numpy()
        goal = obss[0][4:6]
        obss = obss[:, : self.obs_dim]
        utils.visualize_policy_var(self.ts, next(self.rng_seq), self.config, obss, goal)

        if self.config.logger_cls == "vizdom":
            self.plotter.viz.matplot(
                plt, env=self.plotter.viz.env, win=f"viz_ep_{epoch}"
            )
