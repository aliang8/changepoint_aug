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


import os

os.environ["MUJOCO_GL"] = "egl"


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


def load_maze_data(
    data_file: str,
    num_trajs: int,
    batch_size: int,
    train_perc: float = 1.0,
):
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # data is a list of tuples
    # use done to determine when it ends
    rollouts = []
    rollout = []
    for t, transition in enumerate(data):
        obs_t, obs_tp1, action, rew, terminated, truncated, info = transition

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

    for rollout in rollouts[:num_trajs]:
        obs = [step[0] for step in rollout]
        if isinstance(obs[0], np.ndarray):
            obs[0] = obs[0][0]
        obs = [np.concatenate((o["observation"], o["desired_goal"])) for o in obs]
        obs_data.extend(obs[:-1])
        obs_tp1_data.extend(obs[1:])
        action_data.extend([step[1] for step in rollout][:-1])
        action_tp1_data.extend([step[1] for step in rollout][1:])
        done_data.extend([step[3] for step in rollout][1:])
        rew_data.extend([step[2] for step in rollout][1:])

    # convert to torch tensors
    obs_data = torch.from_numpy(np.array(obs_data))
    obs_tp1_data = torch.from_numpy(np.array(obs_tp1_data))
    action_data = torch.from_numpy(np.array(action_data))
    action_tp1_data = torch.from_numpy(np.array(action_tp1_data))
    rew_data = torch.from_numpy(np.array(rew_data))
    done_data = torch.from_numpy(np.array(done_data))

    print("obs_data shape: ", obs_data.shape, "action_data shape: ", action_data.shape)
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


def run_rollout_maze(
    ts,
    num_episodes: int,
    max_episode_steps: int = 1000,
    save_video: bool = False,
    video_dir: str = "videos",
    disable_tqdm: bool = False,
):
    rng_seq = hk.PRNGSequence(0)
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

    all_rewards = []
    all_lengths = []
    num_successes = 0
    rollouts = []

    for rollout_idx in tqdm.tqdm(range(num_episodes), disable=disable_tqdm):
        obs, _ = env.reset()
        done = False
        t = 0
        total_reward = 0

        all_frames = []

        # for maze env
        obs_input = np.concatenate((obs["observation"], obs["desired_goal"]))

        while not done:
            action, _, _ = ts.apply_fn(ts.params, next(rng_seq), obs_input)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs_input = np.concatenate(
                (next_obs["observation"], next_obs["desired_goal"])
            )
            total_reward += reward

            if info["success"]:
                num_successes += 1

            done = done or truncated

            if save_video:
                img = env.render()
                all_frames.append(img)
            t += 1
            rollouts.append((obs, next_obs, action, reward, done, truncated, info))

            obs_input = next_obs_input
            obs = next_obs

            # print(
            #     "t: ", t, "reward: ", reward, "done: ", done, "truncated: ", truncated
            # )

        all_rewards.append(total_reward)
        all_lengths.append(t)

        if save_video:
            imageio.mimsave(
                os.path.join(video_dir, f"rollout_{rollout_idx}.mp4"),
                all_frames,
                fps=30,
            )

    print("average return: ", np.mean(all_rewards))
    print("average success rate: ", num_successes / num_episodes)
    print("average length: ", np.mean(all_lengths))
    return rollouts


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
