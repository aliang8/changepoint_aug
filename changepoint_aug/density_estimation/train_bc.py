from absl import app
from utils import load_maze_data, run_rollout_maze
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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_config():
    config = ConfigDict()
    config.seed = 0
    config.data_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/datasets"
    )
    config.data_file = "sac_maze_100_episodes.pkl"
    config.video_dir = (
        "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/videos"
    )
    config.batch_size = 128
    config.hidden_size = 128
    config.lr = 3e-4
    config.num_epochs = 200
    config.train_perc = 0.9
    config.shuffle_dataset = True
    config.test_interval = 10
    config.num_eval_episodes = 10
    config.max_episode_steps = 1000
    config.num_trajs = 100
    config.results_file = "/scr/aliang80/changepoint_aug/changepoint_aug/online_rl_training/model_ckpts/bc_maze_params.pkl"
    config.mode = "train"
    config.save_rollouts = False
    return config


_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


@hk.transform
def policy_fn(obs, hidden_size, action_dim):
    return GaussianPolicy(hidden_size, action_dim)(obs)


class BCTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config

        # load data
        data_file = os.path.join(config.data_dir, config.data_file)
        _, self.train_dataloader, self.test_dataloader, obs_dim, action_dim = (
            load_maze_data(
                data_file,
                batch_size=config.batch_size,
                num_trajs=config.num_trajs,
                train_perc=config.train_perc,
                shuffle=config.shuffle_dataset,
            )
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.rng_seq = hk.PRNGSequence(config.seed)

        # create
        self.policy_apply = partial(
            jax.jit(policy_fn.apply, static_argnums=(3, 4)),
            hidden_size=config.hidden_size,
            action_dim=action_dim,
        )
        self.ts = self.create_ts(next(self.rng_seq))
        self.jit_update_step = jax.jit(self.update_step)

    def bc_loss_fn(self, params, ts, obss, actions, rng_key):
        _, policy_rng = jax.random.split(rng_key)
        _, action_dist, _ = ts.apply_fn(params, policy_rng, obss)
        logp = action_dist.log_prob(actions)
        return -jnp.mean(logp)

    def update_step(self, ts, rng_key, obss, actions):
        bc_loss, grads = jax.value_and_grad(self.bc_loss_fn)(
            ts.params, ts, obss, actions, rng_key
        )
        ts = ts.apply_gradients(grads=grads)
        return ts, bc_loss

    def create_ts(self, rng_key):
        sample_obs = jnp.zeros((1, self.obs_dim))
        _, init_policy_key = jax.random.split(rng_key)
        params = policy_fn.init(
            init_policy_key, sample_obs, self.config.hidden_size, self.action_dim
        )
        opt = optax.adam(self.config.lr)
        ts = TrainState.create(
            apply_fn=self.policy_apply,
            params=params,
            tx=opt,
        )

        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {param_count}")
        return ts

    def train(self):
        best_loss = float("inf")
        for epoch in tqdm.tqdm(range(self.config.num_epochs)):
            if epoch % self.config.test_interval == 0:
                start = time.time()
                run_rollout_maze(
                    self.ts,
                    num_episodes=self.config.num_eval_episodes,
                    max_episode_steps=self.config.max_episode_steps,
                )
                end = time.time()
                print(
                    f"rollout time: {end - start} for {self.config.num_eval_episodes} episodes"
                )

            epoch_loss = 0
            for batch in self.train_dataloader:
                obss, actions, *_ = batch
                self.ts, batch_bc_loss = self.jit_update_step(
                    self.ts, next(self.rng_seq), obss, actions
                )
                epoch_loss += batch_bc_loss

            epoch_loss /= len(self.train_dataloader)

            if (epoch % self.config.test_interval == 0) and len(
                self.test_dataloader
            ) > 0:
                test_loss = 0
                for batch in self.test_dataloader:
                    obss, actions, *_ = batch
                    batch_bc_loss = self.bc_loss_fn(
                        self.ts.params, self.ts, obss, actions, next(self.rng_seq)
                    )
                    test_loss += batch_bc_loss

                test_loss /= len(self.test_dataloader)
                print(f"epoch: {epoch}, loss: {epoch_loss}, test loss: {test_loss}")

                # save checkpoint
                if test_loss < best_loss:
                    print(f"saving new best model epoch {epoch}, test loss {test_loss}")
                    best_loss = test_loss
                    with open(self.config.results_file, "wb") as f:
                        pickle.dump(self.ts.params, f)

    def eval(self):
        # load from ckpt file
        with open(self.config.results_file, "rb") as f:
            params = pickle.load(f)
            self.ts = self.ts.replace(params=params)

        rollouts = run_rollout_maze(
            self.ts,
            num_episodes=self.config.num_eval_episodes,
            max_episode_steps=self.config.max_episode_steps,
            save_video=True,
            video_dir=self.config.video_dir,
        )

        # save this to dataset
        if self.config.save_rollouts:
            with open(
                f"datasets/bc_policy_rollouts_{self.config.num_eval_episodes}.pkl", "wb"
            ) as f:
                pickle.dump(rollouts, f)


def main(_):
    config = _CONFIG.value
    print(config)
    trainer = BCTrainer(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


if __name__ == "__main__":
    app.run(main)
