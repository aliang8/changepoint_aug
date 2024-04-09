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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
from pathlib import Path
import os
import wandb

# import metaworld
import time

os.environ["MUJOCO_GL"] = "egl"


MAZE_MAPS = {
    "standard": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, "g", "g", "g", "g", "g", 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, "r", "r", "r", "r", "r", 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "all_goals": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
}


def make_env(
    env_name,
    env_id,
    seed,
    num_envs: int = 1,
    maze_map: str = "standard",
    freeze_rand_vec: bool = False,
    goal_observable: bool = True,
    max_episode_steps: int = 1000,
):
    def thunk():
        if env_name == "MW":
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
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif env_name == "MAZE":
            env = gym.make(
                "PointMaze_UMazeDense-v3",
                maze_map=MAZE_MAPS[maze_map],
                continuing_task=False,
                reset_target=False,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
                reward_type="dense",
                camera_id=0,
            )
        return env

    if num_envs == 1:
        env = thunk()
    else:
        env = gym.vector.AsyncVectorEnv([lambda: thunk() for _ in range(num_envs)])

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
    envs = make_env(
        config.env,
        config.env_id,
        config.seed
        + 10000,  # to make sure the evaluation seeds are different from the ones for data collection
        num_envs=config.num_eval_envs,
        max_episode_steps=config.max_episode_steps,
    )

    # import ipdb

    # ipdb.set_trace()

    all_rewards = []
    all_lengths = []
    num_successes = 0
    all_rollouts = []
    all_success_videos = []
    all_failure_videos = []

    render = config.save_video or (wandb_run is not None and config.visualize)

    num_success_save = 5
    num_failure_save = 5

    obs, _ = envs.reset(seed=config.seed)

    rollout_start = time.time()

    for _ in tqdm.tqdm(
        range(int(np.ceil(config.num_eval_episodes // config.num_eval_envs))),
        disable=config.disable_tqdm,
    ):
        obs, _ = envs.reset()

        # print(f"rollout_idx: {rollout_idx}, obs: {obs}")

        # for maze env
        if config.env == "MAZE":
            if "meta" in obs:
                obs_input = np.concatenate(
                    (obs["observation"], obs["desired_goal"], obs["meta"]), axis=-1
                )
            else:
                obs_input = np.concatenate((obs["observation"], obs["desired_goal"]))
        else:
            obs_input = obs

        t = 0
        success = [False for _ in range(config.num_eval_envs)]
        all_frames = [[] for _ in range(config.num_eval_envs)]
        rollouts = [[] for _ in range(config.num_eval_envs)]
        done = [False for _ in range(config.num_eval_envs)]
        total_reward = [0 for _ in range(config.num_eval_envs)]
        episode_len = [0 for _ in range(config.num_eval_envs)]
        all_done = [False for _ in range(config.num_eval_envs)]

        start = time.time()
        while not all(all_done):
            # for ensemble
            if config.num_policies > 1:
                policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
                rng_key = policy_rng_keys[0]
                if config.policy_cls == "mlp":
                    action = jax.vmap(
                        lambda param, rng_key: ts.apply_fn(param, rng_key, obs_input)
                    )(ts.params, policy_rng_keys[1:])
                    action = action.mean(axis=0)
                else:
                    mean, logvar = jax.vmap(
                        lambda param, rng_key: ts.apply_fn(param, rng_key, obs_input)
                    )(ts.params, policy_rng_keys[1:])
                    action = mean.mean(axis=0)  # use mean for evaluation
            else:
                action = ts.apply_fn(ts.param, rng_key, obs_input)

            action = action.squeeze()

            if isinstance(action, jnp.ndarray):
                # env does not work with jax array
                action = action.tolist()

            # import ipdb

            # ipdb.set_trace()
            next_obs, reward, done, truncated, info = envs.step(action)

            if config.env == "MAZE":
                if "meta" in obs:
                    next_obs_input = np.concatenate(
                        (
                            next_obs["observation"],
                            next_obs["desired_goal"],
                            next_obs["meta"],
                        ),
                        axis=-1,
                    )
                else:
                    next_obs_input = np.concatenate(
                        (next_obs["observation"], next_obs["desired_goal"])
                    )
            else:
                next_obs_input = next_obs

            total_reward += reward

            if "final_info" in info:
                for indx, info_ in enumerate(info["final_info"]):
                    if info_ is not None and info_["success"]:
                        success[indx] = True

            if render and (
                len(all_success_videos) < num_success_save
                or len(all_failure_videos) < num_failure_save
            ):
                # img = envs.render()
                imgs = envs.call("render")  # this returns a tuple

                for indx, img in enumerate(imgs):
                    if config.env == "MW":
                        img = np.rot90(img, k=2)

                    if len(all_frames[indx]) < 200:
                        all_frames[indx].append(img)

            for indx, done_ in enumerate(done):
                done_ = done_ or truncated[indx]

                if not all_done[indx]:
                    episode_len[indx] += 1
                    rollouts[indx].append(
                        (
                            {k: v[indx] for k, v in obs.items()},
                            {k: v[indx] for k, v in next_obs.items()},
                            action[indx],
                            reward[indx],
                            done[indx],
                            truncated[indx],
                            {k: v[indx] for k, v in info.items()},
                        )
                    )

                if done_:
                    all_done[indx] = True

            t += 1
            obs_input = next_obs_input
            obs = next_obs

            # print(
            #     "t: ", t, "reward: ", reward, "done: ", done, "truncated: ", truncated
            # )

        # print(f"time for rollout: {time.time() - start}")
        all_rollouts.extend(rollouts)
        num_successes += sum(success)
        all_rewards.extend(total_reward)
        all_lengths.append(episode_len)

        # only add when we did not get enough success / failure clips
        if render and (
            len(all_success_videos) < num_success_save
            or len(all_failure_videos) < num_failure_save
        ):
            for indx, frames in enumerate(all_frames):
                success_ = success[indx]
                frames = np.array(frames)
                # make T x C x H x W
                frames = frames.transpose(0, 3, 1, 2)

                # cap max frames for one trajectory
                if success_ and len(all_success_videos) < num_success_save:
                    all_success_videos.append(frames)

                if not success_ and len(all_failure_videos) < num_failure_save:
                    all_failure_videos.append(frames)

    total_rollout_time = time.time() - rollout_start
    # print(f"total_rollout_time: {total_rollout_time}")

    # import ipdb

    # ipdb.set_trace()

    # combine videos together
    if wandb_run:
        wandb_run.log(
            {
                "time/rollout_time": total_rollout_time,
                "time/avg_rollout_time": total_rollout_time / config.num_eval_episodes,
            }
        )

        if render:
            for label, videos in zip(
                ["success", "failure"], [all_success_videos, all_failure_videos]
            ):
                if len(videos) > 0:
                    max_len = max(len(v) for v in videos)
                    videos = [
                        np.pad(v, ((0, max_len - len(v)), (0, 0), (0, 0), (0, 0)))
                        for v in videos
                    ]
                    videos = np.stack(videos)  # limit number of videos to render
                    wandb_run.log(
                        {f"rollout_{label}": wandb.Video(videos, fps=30, format="mp4")}
                    )

    logging.info(f"average return: {np.mean(all_rewards)}")
    logging.info(f"average success rate: {num_successes / config.num_eval_episodes}")
    logging.info(f"average length: {np.mean(all_lengths)}")

    metrics = {
        "avg_return": np.mean(all_rewards),
        "avg_success_rate": num_successes / config.num_eval_episodes,
        "avg_length": np.mean(all_lengths),
    }
    return rollouts, metrics
