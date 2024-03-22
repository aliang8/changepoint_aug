from absl import logging
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
import wandb
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
            camera_id=0,
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


def run_rollouts(ts, rng_key, config: ConfigDict, wandb_run=None):
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

    all_videos = []
    obs = env.reset(seed=config.seed)

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

            if config.save_video or wandb_run is not None:
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

        if config.save_video or wandb_run is not None:
            all_frames = np.array(all_frames)
            # make T x C x H x W
            all_frames = all_frames.transpose(0, 3, 1, 2)

            # cap max frames for one trajectory
            if len(all_frames) < 500:
                all_videos.append(all_frames)

        if config.save_video:
            imageio.mimsave(
                os.path.join(video_dir, f"rollout_{rollout_idx}.mp4"),
                all_frames,
                fps=30,
            )

    # combine videos together
    if wandb_run:
        if len(all_videos) > 0:
            max_len = max(len(v) for v in all_videos)
            all_videos = [
                np.pad(v, ((0, max_len - len(v)), (0, 0), (0, 0), (0, 0)))
                for v in all_videos
            ]
            all_videos = np.stack(all_videos)
            wandb_run.log({f"rollout": wandb.Video(all_videos, fps=30, format="mp4")})

    logging.info(f"average return: {np.mean(all_rewards)}")
    logging.info(f"average success rate: {num_successes / config.num_eval_episodes}")
    logging.info(f"average length: {np.mean(all_lengths)}")

    metrics = {
        "avg_return": np.mean(all_rewards),
        "avg_success_rate": num_successes / config.num_eval_episodes,
        "avg_length": np.mean(all_lengths),
    }
    return rollouts, metrics
