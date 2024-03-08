from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import jax
import optax
import pickle
import gymnasium as gym
import haiku as hk
import imageio
import tqdm
import jax.numpy as jnp
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from tensordict import TensorDict
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
from pathlib import Path
import os
import metaworld

os.environ["MUJOCO_GL"] = "egl"


def make_env(
    env,
    env_id,
    seed,
    freeze_rand_vec: bool = False,
    goal_observable: bool = True,
    max_episode_steps: int = 1000,
):
    if env == "MW":
        # env_name = env_name.replace("metaworld-", "")
        # env_name = "drawer-open-v2"
        if goal_observable:
            env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
                env_id + "-goal-observable"
            ](seed=seed, render_mode="rgb_array")
        else:
            env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
                env_id + "-goal-hidden"
            ](seed=seed)

        env.camera_name = "corner"  # corner2, corner3, behindGripper
        # to randomize it change the freeze rand vec
        # make sure seeded env has the same goal
        env._freeze_rand_vec = freeze_rand_vec
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.goal_space.seed(seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    elif env == "MAZE":
        # map doesn't matter
        map = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, "g", "g", "g", "g", "g", 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, "r", "r", "r", "r", "r", 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]

        env = gym.make(
            "PointMaze_UMazeDense-v3",
            maze_map=map,
            continuing_task=False,
            reset_target=False,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
            reward_type="dense",
        )
    return env


def create_masks(
    input_size: int,
    hidden_size: int,
    n_hidden: int,
    input_order="sequential",
    input_degrees=None,
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [jnp.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [jnp.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [jnp.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [jnp.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [jnp.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [jnp.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, axis=0))]

    return masks, degrees[0]


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def create_learning_rate_fn(
    num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch
):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, torch.utils.data.default_collate(batch))


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def load_pkl_dataset(
    data_file: str,
    num_trajs: int,
    batch_size: int,
    train_perc: float = 1.0,
    env: str = "MAZE",
):
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # data is a list of tuples
    # use done to determine when it ends
    rollouts = []
    rollout = []
    for t, transition in enumerate(data):
        obs_t, obs_tp1, action, rew, terminated, truncated, info = transition
        if env == "MW":
            terminated = info["success"][0]
        rollout.append((obs_t, action, rew, terminated, truncated, info))

        if terminated or truncated:
            rollouts.append(rollout)
            rollout = []

    print("number of rollouts: ", len(rollouts))
    print("average length of rollout: ", sum(len(r) for r in rollouts) / len(rollouts))

    obs_data = []
    obs_tp1_data = []
    action_data = []
    action_tp1_data = []
    rew_data = []
    done_data = []

    # select random trajectories
    traj_indices = np.random.choice(
        len(rollouts), min(len(rollouts), num_trajs), replace=False
    )
    rollouts = [rollouts[i] for i in traj_indices.tolist()]

    for rollout in rollouts:
        obs = [step[0] for step in rollout]
        if env == "MAZE":
            if isinstance(obs[0], np.ndarray):
                obs[0] = obs[0][0]

            if "meta" in obs[0]:
                obs = [
                    np.concatenate((o["observation"], o["desired_goal"], o["meta"]))
                    for o in obs
                ]
            else:
                obs = [
                    np.concatenate((o["observation"], o["desired_goal"])) for o in obs
                ]
        obs_data.extend(obs[:-1])
        obs_tp1_data.extend(obs[1:])
        action_data.extend([step[1] for step in rollout][:-1])
        action_tp1_data.extend([step[1] for step in rollout][1:])
        done = [step[3] for step in rollout][1:]
        done[-1] = True
        done_data.extend(done)
        rew_data.extend([step[2] for step in rollout][1:])

    # convert to torch tensors
    obs_data = torch.from_numpy(np.array(obs_data)).squeeze()
    obs_tp1_data = torch.from_numpy(np.array(obs_tp1_data)).squeeze()
    action_data = torch.from_numpy(np.array(action_data))
    action_tp1_data = torch.from_numpy(np.array(action_tp1_data))
    rew_data = torch.from_numpy(np.array(rew_data))
    done_data = torch.from_numpy(np.array(done_data))

    # tensor_dict = {
    #     "obs_data": obs_data,
    #     "obs_tp1_data": obs_tp1_data,
    #     "action_data": action_data,
    #     "action_tp1_data": action_tp1_data,
    #     "rew_data": rew_data,
    #     "done_data": done_data,
    # }

    print("obs_data shape: ", obs_data.shape, "action_data shape: ", action_data.shape)
    print(f"min obs data: {obs_data.min(axis=0)}, max obs data: {obs_data.max(axis=0)}")
    print(
        f"min action data: {action_data.min(axis=0)}, max action data: {action_data.max(axis=0)}"
    )

    # data = TensorDict(tensor_dict, batch_size=len(obs_data))

    # create dataloader
    dataset = torch.utils.data.TensorDataset(
        obs_data, action_data, obs_tp1_data, action_tp1_data, rew_data, done_data
    )

    # split into train and test
    train_size = int(train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # convert for using with jax
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = NumpyLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    print("number of train batches: ", len(train_dataloader))
    print("number of test batches: ", len(test_dataloader))

    return (
        dataset,
        train_dataloader,
        test_dataloader,
        obs_data.shape[-1],
        action_data.shape[-1],
    )


def run_rollouts(ts, rng_key, config: ConfigDict):
    env = make_env(
        config.env,
        config.env_id,
        config.seed,
        max_episode_steps=config.max_episode_steps,
    )

    all_rewards = []
    all_lengths = []
    num_successes = 0
    rollouts = []

    video_dir = Path(config.root_dir) / config.video_dir

    for rollout_idx in tqdm.tqdm(
        range(config.num_eval_episodes), disable=config.disable_tqdm
    ):
        obs, _ = env.reset()
        done = False
        t = 0
        total_reward = 0

        all_frames = []

        # for maze env
        if config.env == "MAZE":
            if "meta" in obs:
                obs_input = np.concatenate(
                    (obs["observation"], obs["desired_goal"], obs["meta"])
                )
            else:
                obs_input = np.concatenate((obs["observation"], obs["desired_goal"]))
        else:
            obs_input = obs

        success = False

        while not done:
            # for ensemble
            if config.num_policies > 1:
                policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
                rng_key = policy_rng_keys[0]
                if config.policy_cls == "mlp":
                    action = jax.vmap(
                        lambda param, rng_key: ts.apply_fn(
                            param, rng_key, obs_input[None]
                        )
                    )(ts.params, policy_rng_keys[1:])
                else:
                    action, _, mean, stddev = jax.vmap(
                        lambda param, rng_key: ts.apply_fn(
                            param, rng_key, obs_input[None]
                        )
                    )(ts.params, policy_rng_keys[1:])
                action = action.mean(axis=0)
            else:
                action = ts.apply_fn(ts.param, rng_key, obs_input)

            action = action.squeeze()

            if isinstance(action, jnp.ndarray) and config.env == "MW":
                # metaworld does not work with jax ndarray
                action = action.tolist()

            next_obs, reward, done, truncated, info = env.step(action)

            if config.env == "MAZE":
                if "meta" in obs:
                    next_obs_input = np.concatenate(
                        (
                            next_obs["observation"],
                            next_obs["desired_goal"],
                            next_obs["meta"],
                        )
                    )
                else:
                    next_obs_input = np.concatenate(
                        (next_obs["observation"], next_obs["desired_goal"])
                    )
            else:
                next_obs_input = next_obs

            total_reward += reward

            if info["success"]:
                success = True

            done = done or truncated

            if config.save_video:
                img = env.render()
                all_frames.append(img)
            t += 1
            rollouts.append((obs, next_obs, action, reward, done, truncated, info))

            obs_input = next_obs_input
            obs = next_obs

            # print(
            #     "t: ", t, "reward: ", reward, "done: ", done, "truncated: ", truncated
            # )

        if success:
            num_successes += 1

        all_rewards.append(total_reward)
        all_lengths.append(t)

        if config.save_video:
            imageio.mimsave(
                os.path.join(video_dir, f"rollout_{rollout_idx}.mp4"),
                all_frames,
                fps=30,
            )

    print("average return: ", np.mean(all_rewards))
    print("average success rate: ", num_successes / config.num_eval_episodes)
    print("average length: ", np.mean(all_lengths))

    metrics = {
        "average_return": np.mean(all_rewards),
        "average_success_rate": num_successes / config.num_eval_episodes,
        "average_length": np.mean(all_lengths),
    }
    return rollouts, metrics


def make_blob_dataset(
    n_samples, centers, n_features, random_state, train_perc: float, batch_size: int
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(X, y)

    # split into train and test
    train_size = int(train_perc * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # convert for using with jax
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = NumpyLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    print("number of train batches: ", len(train_dataloader))
    print("number of test batches: ", len(test_dataloader))

    return (
        dataset,
        train_dataloader,
        test_dataloader,
        2,
        1,
    )


def visualize_policy_var(ts, rng_key, config, obss, goal):
    T = np.arange(len(obss))

    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
    action_preds = jax.vmap(lambda param, rng_key: ts.apply_fn(param, rng_key, obss))(
        ts.params, policy_rng_keys[1:]
    )
    # compute variance between ensemble
    variance = jnp.var(action_preds, axis=0)

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
