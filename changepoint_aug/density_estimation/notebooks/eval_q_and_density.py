import jax
import sys

sys.path.append("/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation")
import pickle
import jax.numpy as jnp
import os
import haiku as hk
import matplotlib.animation as animation
from utils import load_maze_data
import pickle
import jax
import torch
import haiku as hk
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from models import q_fn, vae_fn, decode_fn, get_prior_fn
from functools import partial
from changepoint_aug.density_estimation.configs.q_sarsa import (
    get_config as get_q_config,
)
from changepoint_aug.density_estimation.configs.vae import get_config as get_vae_config
from tensorflow_probability.substrates import jax as tfp
from pprint import pprint

dist = tfp.distributions
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == "__main__":
    # get arbitary config
    q_config = get_q_config()
    rng_key = jax.random.PRNGKey(0)

    data_file = os.path.join(q_config.data_dir, "sac_maze_500_episodes.pkl")

    # load trained q model
    q_model_file = os.path.join(q_config.results_dir, "q_params.pkl")
    with open(q_model_file, "rb") as f:
        q_params = pickle.load(f)

    vae_config = get_vae_config()

    # load vae density model
    gc_vae_model_file = os.path.join(
        vae_config.results_dir, "vae_params_maze_xy_obs.pkl"
    )
    with open(gc_vae_model_file, "rb") as f:
        vae_params = pickle.load(f)

    pprint(vae_config)

    print("loading q model from", q_model_file)
    print("loading vae model from", gc_vae_model_file)

    # load dataset
    dataset, train_dataloader, test_dataloader, obs_dim, action_dim = load_maze_data(
        data_file,
        batch_size=q_config.batch_size,
        num_trajs=q_config.num_trajs,
        train_perc=q_config.train_perc,
    )

    apply_q_fn = partial(
        hk.without_apply_rng(q_fn).apply, hidden_size=q_config.hidden_size
    )
    apply_vae_fn = partial(
        vae_fn.apply,
        latent_size=vae_config.latent_size,
        hidden_size=vae_config.hidden_size,
        obs_dim=2,
        cond_dim=2,
    )
    apply_decode_fn = partial(
        hk.without_apply_rng(decode_fn).apply,
        latent_size=vae_config.latent_size,
        hidden_size=vae_config.hidden_size,
        obs_dim=2,
        cond_dim=2,
    )
    apply_get_prior_fn = partial(
        hk.without_apply_rng(get_prior_fn).apply,
        latent_size=vae_config.latent_size,
        hidden_size=vae_config.hidden_size,
    )

    def estimate_density(vae_params, rng_key, obs, goal, kl_div_weight, latent_size):
        vae_output = apply_vae_fn(vae_params, rng_key, obs, goal)

        # sample a bunch of z's from the posterior and decode them
        # posterior is a normal distribution with mean and stddev
        # z = dist.Normal(0, 1)
        z = apply_get_prior_fn(vae_params, goal)
        num_posterior_samples = 100
        z_samples = z.sample(seed=rng_key, sample_shape=(num_posterior_samples,))
        # jax.debug.print(f"{z_samples.shape}")

        # decode conditioned on goal
        obs_pred = jax.vmap(lambda z: apply_decode_fn(vae_params, z, goal))(z_samples)

        # repeat obs for each z sample
        obs = jnp.repeat(obs[None], num_posterior_samples, axis=0)

        # compute average l2 loss
        recon_loss = optax.squared_error(obs_pred, obs).sum(axis=-1).mean()
        # jax.debug.print(f"{recon_loss.shape}")

        prior = dist.Normal(0, 1)
        posterior = dist.Normal(loc=vae_output.mean, scale=vae_output.stddev)
        kl_div_loss = dist.kl_divergence(posterior, prior).sum(axis=-1).mean()
        # jax.debug.breakpoint()

        # jax.debug.print(f"{kl_div_loss}")
        # jax.debug.print(f"{recon_loss}")

        # we want to maximize elbo
        # elbo = E[log p(x|z)] - KL[q(z|x) || p(z)]
        loss = recon_loss + kl_div_weight * kl_div_loss
        elbo = -loss
        return elbo

    for traj_indx in range(100, 110):
        start_indices = np.where(dataset[:][-1])[0]
        start_indices += 1
        start_indices = np.insert(start_indices, 0, 0)

        start = start_indices[traj_indx]
        end = start_indices[traj_indx + 1]

        (
            all_obss,
            all_actions,
            all_next_obss,
            all_next_actions,
            all_rewards,
            all_dones,
        ) = dataset[:]

        min_x = torch.min(all_obss[:, 0])
        max_x = torch.max(all_obss[:, 0])
        min_y = torch.min(all_obss[:, 1])
        max_y = torch.max(all_obss[:, 1])

        obss = all_obss[start:end].numpy()
        actions = all_actions[start:end].numpy()
        rewards = all_rewards[start:end].numpy()

        print(obss.shape, actions.shape, rewards.shape)

        T = np.arange(len(obss))

        # compute q_values along that trajectory
        q_values = apply_q_fn(q_params, obss[:, :2], actions)
        q_values = jnp.squeeze(q_values, axis=-1)
        print(q_values.shape)

        plt.clf()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # first subplot plot the q_values and rewards
        axes[0].plot(T, q_values, label="Q-value", linewidth=4)
        axes[0].plot(T, rewards, label="Reward", linewidth=4)
        axes[0].legend()
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Q-value")
        axes[0].set_title(f"Q-values for rollout {traj_indx}")

        # compute metric: var(Q(s, a+gaussian noise))
        num_samples = 100

        def apply_noise(params, state, action):
            noise_samples = jax.random.uniform(
                rng_key, (num_samples, 2), minval=-0.1, maxval=0.1
            )
            q = jax.vmap(lambda noise: apply_q_fn(params, state, action + noise))(
                noise_samples
            )
            return q

        def apply_action(params, states, actions):
            q = jax.vmap(lambda state, action: apply_noise(params, state, action))(
                states, actions
            )
            return q

        q = apply_action(q_params, obss[:, :2], actions)
        q = jnp.squeeze(q, axis=-1)
        # [T, noise_samples]
        variance = jnp.var(q, axis=1)
        print(variance.shape)

        axes[1].plot(T, variance, linewidth=4)
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Q-value variance")
        axes[1].set_title("Variance of Q-values")

        # plot the trajectory
        traj_xy = obss[:, :2]
        (traj_plot,) = axes[2].plot(traj_xy[:, 0], traj_xy[:, 1], linewidth=4)
        goal_location = dataset[start][0][-2:]
        axes[2].plot(goal_location[0], goal_location[1], "g*", markersize=10)
        axes[2].set_xlim(min_x, max_x)
        axes[2].set_ylim(min_y, max_y)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title(f"Trajectory {traj_indx}")

        # find location with max variance
        max_var_idx = np.argmax(variance)
        # set vline there
        axes[0].axvline(x=max_var_idx, color="r", linestyle="--", linewidth=4)
        axes[1].axvline(x=max_var_idx, color="r", linestyle="--", linewidth=4)
        # make a point in axes[2]
        max_var_point = obss[max_var_idx, :2]
        axes[2].plot(max_var_point[0], max_var_point[1], "ro")

        # plot all of the trajectorie
        all_obss = dataset[:][0].numpy()
        all_goal_locations = all_obss[:, -2:]
        all_start_locations = all_obss[start_indices[:-1], :2]
        axes[3].plot(all_obss[:, 0], all_obss[:, 1])
        axes[3].plot(
            all_goal_locations[:, 0], all_goal_locations[:, 1], "g*", markersize=10
        )
        axes[3].plot(
            all_start_locations[:, 0], all_start_locations[:, 1], "bo", markersize=10
        )
        axes[3].set_title("All trajectories + start and goal locations")

        # plot the density of states conditioned on goal
        sample_goal = all_goal_locations[start_indices[90]]
        rng_keys = jax.random.split(rng_key, all_obss.shape[0])
        density = jax.vmap(
            lambda obs, key: estimate_density(
                vae_params,
                key,
                obs,
                sample_goal,
                vae_config.kl_div_weight,
                vae_config.latent_size,
            )
        )(all_obss[:, :2], rng_keys)
        print(f"min density: {np.min(density)}, max density: {np.max(density)}")

        scat = axes[4].scatter(
            all_obss[:, 0], all_obss[:, 1], c=density, cmap="viridis"
        )
        axes[4].plot(sample_goal[0], sample_goal[1], "g*", markersize=10)
        axes[4].set_title("GC Density, Goal: (x, y) = " + str(sample_goal))
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")
        axes[4].set_xlim(min_x, max_x)
        axes[4].set_ylim(min_y, max_y)
        fig.colorbar(scat, ax=axes[4])

        plt.savefig(f"imgs/density_traj_{traj_indx}.png")
