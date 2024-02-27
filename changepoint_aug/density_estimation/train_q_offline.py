from utils import load_maze_data, run_rollout_maze
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from models import QFunction, Policy
import os
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
from functools import partial
from flax.training.train_state import TrainState
from typing import Any

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_config():
    config = ConfigDict()
    config.seed = 0
    config.data_file = "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/sac_maze_100_episodes.pkl"
    config.video_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/videos"
    )
    config.batch_size = 128
    config.hidden_size = 128
    config.lr = 3e-4
    config.num_epochs = 200
    config.train_perc = 0.9
    config.test_interval = 10
    config.num_eval_episodes = 10
    config.max_episode_steps = 1000
    config.num_trajs = 100
    config.results_file = "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/model_ckpts/bc_maze_params.pkl"
    config.mode = "train"

    # qfunction
    config.gamma = 0.99
    config.tau = 0.005
    return config


@hk.transform
def policy_fn(obs, hidden_size, action_dim):
    return Policy(hidden_size, action_dim)(obs)


@hk.transform
def q_fn(obs, action, hidden_size):
    return QFunction(hidden_size)(obs, action)


class QTrainState(TrainState):
    tx_actor: optax.GradientTransformation
    q_params: flax.core.frozen_dict.FrozenDict
    q_opt_state: flax.core.frozen_dict.FrozenDict
    actor_params: flax.core.frozen_dict.FrozenDict
    actor_opt_state: flax.core.frozen_dict.FrozenDict


class QTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config

        # load data
        _, self.train_dataloader, self.test_dataloader, obs_dim, action_dim = (
            load_maze_data(
                config.data_file,
                batch_size=config.batch_size,
                num_trajs=config.num_trajs,
                train_perc=config.train_perc,
            )
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.rng_seq = hk.PRNGSequence(config.seed)

        # create
        self.q_fn_apply = partial(
            jax.jit(hk.without_apply_rng(q_fn).apply, static_argnums=(3)),
            hidden_size=config.hidden_size,
        )
        self.actor_fn_apply = partial(
            jax.jit(hk.without_apply_rng(policy_fn).apply, static_argnums=(2, 3)),
            hidden_size=config.hidden_size,
            action_dim=action_dim,
        )
        self.ts = self.create_ts(next(self.rng_seq))
        self.jit_update_q_step = jax.jit(self.update_q_step)
        self.jit_update_actor_step = jax.jit(self.update_actor_step)

    def td_loss_fn(
        self, q_params, actor_params, obss, actions, obss_tp1, rewards, dones
    ):
        # Q(s,a) -  r + gamma * (1 - done) * Q(s', pi(s'))
        q = self.q_fn_apply(q_params, obss, actions)
        q = jnp.squeeze(q, axis=-1)
        action_tp1 = self.actor_fn_apply(actor_params, obss_tp1)
        q_tp1 = self.q_fn_apply(q_params, obss_tp1, action_tp1)
        q_tp1 = jnp.squeeze(q_tp1, axis=-1)
        backup = rewards + self.config.gamma * (1 - dones) * q_tp1
        loss_q = optax.squared_error(q, backup).mean()
        return loss_q

    def actor_loss_fn(self, actor_params, q_params, obs):
        # maximize q value
        # E[Q(s, pi(s))]
        policy_action = self.actor_fn_apply(actor_params, obs)
        q = self.q_fn_apply(q_params, obs, policy_action)
        q = jnp.squeeze(q, axis=-1)
        # jax.debug.breakpoint()
        # maybe add entropy
        return -q.mean()

    def update_q_step(self, obss, actions, obss_tp1, rewards, dones):
        td_loss, grads = jax.value_and_grad(self.td_loss_fn)(
            self.ts.q_params,
            self.ts.actor_params,
            obss,
            actions,
            obss_tp1,
            rewards,
            dones,
        )
        updates, q_opt_state = self.ts.tx.update(grads, self.ts.q_opt_state)
        new_params = optax.apply_updates(self.ts.q_params, updates)
        return new_params, q_opt_state, td_loss

    def update_actor_step(self, obss):
        actor_loss, grads = jax.value_and_grad(self.actor_loss_fn)(
            self.ts.actor_params, self.ts.q_params, obss
        )
        updates, actor_opt_state = self.ts.tx_actor.update(
            grads, self.ts.actor_opt_state
        )
        new_params = optax.apply_updates(self.ts.actor_params, updates)
        return new_params, actor_opt_state, actor_loss

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

        train_state = QTrainState.create(
            apply_fn=self.q_fn_apply,
            params=q_params,
            tx=q_opt,
            tx_actor=actor_opt,
            q_params=q_params,
            q_opt_state=q_opt_state,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
        )
        return train_state

    def train(self):
        best_loss = float("inf")
        for epoch in tqdm.tqdm(range(self.config.num_epochs)):
            epoch_q_loss = 0
            epoch_actor_loss = 0
            for batch in self.train_dataloader:
                obss, actions, obss_tp1, _, rewards, dones = batch
                # update q-function
                q_params, q_opt_state, batch_q_loss = self.jit_update_q_step(
                    obss, actions, obss_tp1, rewards, dones
                )
                self.ts = self.ts.replace(q_params=q_params, q_opt_state=q_opt_state)
                epoch_q_loss += batch_q_loss

                # update actor
                actor_params, actor_opt_state, batch_actor_loss = (
                    self.jit_update_actor_step(obss)
                )
                self.ts = self.ts.replace(
                    actor_params=actor_params, actor_opt_state=actor_opt_state
                )
                epoch_actor_loss += batch_actor_loss

            epoch_q_loss /= len(self.train_dataloader)
            epoch_actor_loss /= len(self.train_dataloader)

            if (epoch % self.config.test_interval == 0) and len(
                self.test_dataloader
            ) > 0:
                test_q_loss = 0
                test_actor_loss = 0
                for batch in self.test_dataloader:
                    obss, actions, obss_tp1, _, rewards, dones = batch
                    batch_q_loss = self.td_loss_fn(
                        self.ts.q_params,
                        self.ts.actor_params,
                        obss,
                        actions,
                        obss_tp1,
                        rewards,
                        dones,
                    )
                    test_q_loss += batch_q_loss
                    batch_actor_loss = self.actor_loss_fn(
                        self.ts.actor_params, self.ts.q_params, obss
                    )
                    test_actor_loss += batch_actor_loss

                test_actor_loss /= len(self.test_dataloader)
                test_actor_loss /= len(self.test_dataloader)
                print(
                    f"epoch: {epoch}, q loss: {epoch_q_loss}, actor loss: {epoch_actor_loss} test q loss: {test_q_loss} test actor loss: {test_actor_loss}",
                )

                # save checkpoint
                # if test_loss < best_loss:
                #     print(f"saving new best model epoch {epoch}, test loss {test_loss}")
                #     best_loss = test_loss
                #     with open(self.config.results_file, "wb") as f:
                #         pickle.dump(self.params, f)


if __name__ == "__main__":
    config = get_config()
    trainer = QTrainer(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()
