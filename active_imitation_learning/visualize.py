import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


def visualize_policy_var(ts, rng_key, config, obss, goal):
    T = np.arange(len(obss))

    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
    if config.policy_cls == "mlp":
        action_preds = jax.vmap(
            lambda param, rng_key: ts.apply_fn(param, rng_key, obss)
        )(ts.params, policy_rng_keys[1:])

        # compute variance between ensemble
        variance = jnp.var(action_preds, axis=0)

    else:
        mean, logvar = jax.vmap(
            lambda param, rng_key: ts.apply_fn(param, rng_key, obss)
        )(ts.params, policy_rng_keys[1:])

        # compute variance based on this https://arxiv.org/pdf/1612.01474.pdf
        variance = jnp.exp(logvar)
        ensemble_mean = jnp.mean(mean, axis=0)
        variance = (variance + mean**2).sum(
            axis=0
        ) / config.num_policies - ensemble_mean**2

    # compute mean over action dimension
    variance = jnp.mean(variance, axis=-1)

    # first subplot plot the q_values and rewards
    axes[0].plot(T, variance, label="Policy Ensemble Variance", linewidth=4)
    axes[0].legend()
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Ensemble Variance")
    axes[0].set_title(f"Policy Ensemble Variance")

    # plot the trajectory
    traj_xy = obss[:, :2]
    (traj_plot,) = axes[1].plot(traj_xy[:, 0], traj_xy[:, 1], linewidth=4)
    axes[1].plot(goal[0], goal[1], "g*", markersize=10)
    # axes[2].set_xlim(min_x, max_x)
    # axes[2].set_ylim(min_y, max_y)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"Trajectory")

    # find location with max variance
    max_var_idx = np.argmax(variance)
    # set vline there
    axes[0].axvline(x=max_var_idx, color="r", linestyle="--", linewidth=4)
    max_var_point = obss[max_var_idx, :2]
    axes[1].plot(max_var_point[0], max_var_point[1], "ro")
    return fig


def visualize_q_trajectory(ts, env, rng_key, obss, actions, rewards, goal):
    T = np.arange(len(obss))

    plt.clf()
    num_plots = 2 if env == "MW" else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 6))
    axes = axes.flatten()

    # compute q_values along that trajectory
    q_values = ts.apply_fn(ts.params, obss, actions)
    q_values = jnp.squeeze(q_values, axis=-1)
    # first subplot plot the q_values and rewards
    axes[0].plot(T, q_values, label="Q-value", linewidth=4)
    axes[0].plot(T, rewards, label="Reward", linewidth=4)
    axes[0].legend()
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Q-value")
    axes[0].set_title(f"Q-values for rollout")

    # compute metric: var(Q(s, a+gaussian noise))
    num_samples = 100

    def apply_noise(params, state, action):
        noise_samples = jax.random.uniform(
            rng_key, (num_samples, actions.shape[-1]), minval=-0.1, maxval=0.1
        )
        q = jax.vmap(lambda noise: ts.apply_fn(params, state, action + noise))(
            noise_samples
        )
        return q

    def apply_action(params, states, actions):
        q = jax.vmap(lambda state, action: apply_noise(params, state, action))(
            states, actions
        )
        return q

    q = apply_action(ts.params, obss, actions)
    q = jnp.squeeze(q, axis=-1)
    # [T, noise_samples]
    variance = jnp.var(q, axis=1)
    # print(variance)
    # print(variance.shape)

    axes[1].plot(T, variance, linewidth=4)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Q-value variance")
    axes[1].set_title("Variance of Q-values")

    # find location with max variance
    max_var_idx = np.argmax(variance)
    print("max var timestep:", max_var_idx)
    # set vline there
    axes[0].axvline(x=max_var_idx, color="r", linestyle="--", linewidth=4)
    axes[1].axvline(x=max_var_idx, color="r", linestyle="--", linewidth=4)

    # plot the trajectory
    if env == "MAZE":
        traj_xy = obss[:, :2]
        (traj_plot,) = axes[2].plot(traj_xy[:, 0], traj_xy[:, 1], linewidth=4)
        axes[2].plot(goal[0], goal[1], "g*", markersize=10)
        # axes[2].set_xlim(min_x, max_x)
        # axes[2].set_ylim(min_y, max_y)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title(f"Trajectory")

        # make a point in axes[2]
        max_var_point = obss[max_var_idx, :2]
        axes[2].plot(max_var_point[0], max_var_point[1], "ro")

    return fig


def estimate_density(ts, rng_key, obs, goal, kl_div_weight, num_posterior_samples):
    rng_keys = jax.random.split(rng_key, num_posterior_samples)
    vae_output = jax.vmap(lambda key: ts.apply_fn(ts.params, key, obs, goal))(rng_keys)

    obs_pred = vae_output.recon

    # jax.debug.breakpoint()
    # repeat obs for each z sample
    obs = jnp.repeat(obs[None], num_posterior_samples, axis=0)

    # compute average l2 loss
    recon_loss = optax.squared_error(obs_pred, obs).sum(axis=-1).mean()

    prior = dist.Normal(0, 1)
    posterior = dist.Normal(loc=vae_output.mean, scale=vae_output.stddev)
    kl_div_loss = dist.kl_divergence(posterior, prior).sum(axis=-1).mean()

    # we want to maximize elbo
    # elbo = E[log p(x|z)] - KL[q(z|x) || p(z)]
    loss = recon_loss + kl_div_weight * kl_div_loss
    elbo = -loss
    return elbo


def visualize_density_estimate(ts, rng_key, config, obss, goals, kl_div_weight):
    # visualize density of all trajectories for a few different goals
    # import ipdb

    # ipdb.set_trace()
    plt.clf()
    num_plots = len(goals)
    num_cols = np.ceil(num_plots / 2).astype(int)
    num_rows = num_plots // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 10))
    axes = axes.flatten()

    for goal_indx, goal in enumerate(goals):
        rng_keys = jax.random.split(rng_key, obss.shape[0] + 1)
        rng_key = rng_keys[0]

        # chunk obs into size 1000 to fit on device
        obss_chunked = np.array_split(obss, obss.shape[0] // 1000)
        rng_keys_chunked = np.array_split(rng_keys[1:], obss.shape[0] // 1000)

        densities = []

        for indx, chunk_obs in enumerate(obss_chunked):
            density = jax.vmap(
                lambda obs, key: estimate_density(
                    ts,
                    key,
                    obs,
                    goal,
                    kl_div_weight,
                    config.num_posterior_samples,
                )
            )(chunk_obs[:, :2], rng_keys_chunked[indx])

            densities.append(density)

        density = np.concatenate(densities)
        # print(f"min density: {np.min(density)}, max density: {np.max(density)}")

        scat = axes[goal_indx].scatter(
            obss[:, 0], obss[:, 1], c=density, cmap="viridis"
        )
        axes[goal_indx].plot(
            goal[0], goal[1], color="orange", marker="*", markersize=30
        )
        axes[goal_indx].set_title("GC Density, Goal: (x, y) = " + str(goal))
        axes[goal_indx].set_xlabel("x")
        axes[goal_indx].set_ylabel("y")

    fig.colorbar(scat, ax=axes[4])
    return fig
