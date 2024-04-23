from absl import logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import io
import wandb
from PIL import Image
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
from functools import partial
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp
import active_imitation_learning.utils.utils as utils
import matplotlib.pyplot as plt
from collections import Counter

from active_imitation_learning.trainers.base_trainer import BaseTrainer
from active_imitation_learning.models import policy_fn
from active_imitation_learning.visualize import visualize_policy_var
from active_imitation_learning.utils.env_utils import make_env
from active_imitation_learning.utils.rollout import run_rollouts

dist = tfp.distributions


def create_ts(config, obs_shape, action_dim, rng_key):
    sample_obs = jnp.zeros((1, *obs_shape))

    if config.load_from_ckpt != "":
        params = pickle.load(open(config.load_from_ckpt, "rb"))
    else:
        policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
        params = jax.vmap(policy_fn.init, in_axes=(0, None, None, None, None))(
            policy_rng_keys[1:],
            config.policy_cls,
            sample_obs,
            config.hidden_size,
            action_dim,
        )

    opt = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(config.lr),
    )

    # create
    policy_apply = partial(
        jax.jit(policy_fn.apply, static_argnums=(2, 4, 5)),
        policy_cls=config.policy_cls,
        hidden_size=config.hidden_size,
        action_dim=action_dim,
    )

    ts = TrainState.create(
        apply_fn=policy_apply,
        params=params,
        tx=opt,
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logging.info(f"Number of policy parameters: {param_count}")
    return ts


class BCTrainer(BaseTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)
        # create envs
        self.eval_envs = make_env(
            config.env,
            config.env_id,
            config.seed
            + 10000,  # to make sure the evaluation seeds are different from the ones for data collection
            num_envs=config.num_eval_envs,
            max_episode_steps=config.max_episode_steps,
        )

        self.ts = create_ts(config, self.obs_shape, self.action_dim, next(self.rng_seq))
        self.jit_update_step = jax.jit(self.update_step)

    def bc_loss_fn(self, params, ts, obss, actions, rng_key):

        # import ipdb

        # ipdb.set_trace()
        policy_rng_keys = jax.random.split(rng_key, self.config.num_policies + 1)

        if self.config.policy_cls == "mlp":
            action_pred = jax.vmap(
                lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obss)
            )(params, policy_rng_keys[1:])
            # jax.debug.breakpoint()
            # repeat actions for each policy
            actions = jnp.repeat(actions[None], self.config.num_policies, axis=0)
            bc_loss = optax.squared_error(action_pred, actions).mean()
        elif self.config.policy_cls == "gaussian":
            mean, logvar = jax.vmap(
                lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obss)
            )(params, policy_rng_keys[1:])

            stddev = jnp.exp(logvar) ** 0.5
            action_dist = dist.Normal(mean, stddev)
            logp = action_dist.log_prob(actions)
            logp = logp.sum(axis=-1).mean()
            bc_loss = -logp
        metrics = {"bc_loss": bc_loss}
        return bc_loss, metrics

    def update_step(self, ts, rng_key, obss, actions):
        logging.info("update step")
        (bc_loss, metrics), grads = jax.value_and_grad(self.bc_loss_fn, has_aux=True)(
            ts.params, ts, obss, actions, rng_key
        )
        ts = ts.apply_gradients(grads=grads)
        return ts, bc_loss, metrics

    def train_step(self, batch):
        obss, actions, *_ = batch
        # obss = obss[:, : self.obs_dim]
        # print(obss.shape, actions.shape)

        # update model
        self.ts, _, metrics = self.jit_update_step(
            self.ts, next(self.rng_seq), obss, actions
        )
        return metrics

    def test(self, epoch):
        avg_test_metrics = Counter()
        for batch in self.test_loader:
            obss, actions, *_ = batch
            bc_loss, metrics = self.bc_loss_fn(
                self.ts.params, self.ts, obss, actions, next(self.rng_seq)
            )
            avg_test_metrics += Counter(metrics)

        for k in avg_test_metrics:
            avg_test_metrics[k] /= len(self.test_loader)

        test_metrics = {f"test/{k}": v for k, v in avg_test_metrics.items()}

        # run eval rollouts on preset initial conditions
        if self.config.run_ic_evals:
            logging.info("running eval on preset ics")
            eval_rollouts, rollout_metrics = run_rollouts(
                self.ts,
                rng_key=next(self.rng_seq),
                config=self.config,
                wandb_run=self.wandb_run,
                ics=self.initial_conditions,
            )
            if self.wandb_run:
                rollout_metrics = {
                    f"ic_rollouts/{k}": v for k, v in rollout_metrics.items()
                }
                test_metrics.update(rollout_metrics)

        # run rollouts to test trained policy
        logging.info("running eval on generalization")
        eval_rollouts, rollout_metrics = run_rollouts(
            self.ts,
            rng_key=next(self.rng_seq),
            config=self.config,
            wandb_run=self.wandb_run,
            env=self.eval_envs,
        )

        if self.wandb_run:
            rollout_metrics = {f"rollouts/{k}": v for k, v in rollout_metrics.items()}
            test_metrics.update(rollout_metrics)

        # save the trajectories to file
        metadata_file = self.log_dir / f"eval_rollouts_{epoch}.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(eval_rollouts, f)

        # visualize
        # visualize variance over policy ensemble for random trajectory
        if self.dataset["dones"].sum() == 0:
            start_indices = np.arange(
                0,
                self.dataset["observations"].shape[0],
                self.config.max_episode_steps,
            )
        else:
            start_indices = np.where(self.dataset["dones"])[0]
            start_indices += 1
            start_indices = np.insert(start_indices, 0, 0)
            start_indices = start_indices[:-1]

        # import ipdb

        # ipdb.set_trace()

        # select random trajectory
        traj_indx = np.random.randint(0, len(start_indices) - 1)
        start = start_indices[traj_indx]
        end = start_indices[traj_indx + 1]
        obss = self.dataset["observations"][start:end].numpy()
        if self.config.env == "MAZE":
            goal = obss[0][4:6]
        else:
            goal = None
        # obss = obss[:, : self.obs_dim]
        fig = visualize_policy_var(self.ts, next(self.rng_seq), self.config, obss, goal)

        if self.wandb_run:
            # save as image instead of plotly interactive figure
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            wandb.log(({"viz/policy_var_trajectory": wandb.Image(Image.open(buf))}))

        return test_metrics
