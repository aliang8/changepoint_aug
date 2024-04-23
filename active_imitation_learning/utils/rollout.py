from absl import logging
import numpy as np
import torch
import jax
import optax
import pickle
import gym as old_gym
import gymnasium as gym
import haiku as hk
import imageio
import tqdm
import os
import time
import wandb
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict

from active_imitation_learning.utils.env_utils import make_env

os.environ["MUJOCO_GL"] = "egl"


def run_rollouts(ts, rng_key, config, env=None, wandb_run=None, ics=None):

    if ics is None:
        num_evals = config.num_eval_episodes
        num_iters = int(np.ceil(num_evals / config.num_eval_envs))

        all_rollouts = []
        all_metrics = []

        for indx in tqdm.tqdm(range(num_iters), desc="running eval"):
            save_frames = (
                config.visualize_rollouts
                and (indx * config.num_eval_envs) < config.num_videos_save
            )
            rollouts, metrics = async_rollout_fn(
                ts, rng_key, config, env, save_frames=save_frames
            )
            all_rollouts.append(rollouts)
            all_metrics.append(metrics)

        all_rollouts = {k: [r[k] for r in all_rollouts] for k in rollouts}
        videos = []
        for frames in all_rollouts["frames"]:
            if isinstance(frames, np.ndarray):
                videos.append(frames)

        # each element is [N, T, H, W, C]
        if config.visualize_rollouts:
            all_rollouts["frames"] = np.concatenate(videos, axis=0)

        all_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in metrics}
        label = "eval"
    else:
        all_rollouts, all_metrics = run_rollout_on_ic(ts, rng_key, config, ics)

        if config.visualize_rollouts:
            videos = []
            for frames in all_rollouts["frames"]:
                if len(frames) > 0:
                    videos.append(frames)
            all_rollouts["frames"] = videos

        label = "ic"

    # combine videos together
    if wandb_run:
        wandb_run.log(all_metrics)
        videos = all_rollouts["frames"]
        if config.visualize_rollouts and len(videos) > 0:
            max_len = max(len(v) for v in videos)
            videos = [
                np.pad(v, ((0, max_len - len(v)), (0, 0), (0, 0), (0, 0)))
                for v in videos
            ]
            # N,T,H,W,C -> N,T,C,H,W
            videos = np.array(videos).transpose(0, 1, 4, 2, 3)

            # limit number of videos to render
            wandb_run.log(
                {f"rollout_{label}": wandb.Video(videos, fps=30, format="mp4")}
            )

    return all_rollouts, all_metrics


def run_rollout_on_ic(ts, rng_key, config, ics: List[np.ndarray]):
    rollouts = []
    successes = []

    rollout_start = time.time()
    logging.info(f"number of ICs: {len(ics)}")
    for ic in tqdm.tqdm(ics, desc="running eval rollouts with ICs"):
        # reset env with ic
        env = make_env(
            config.env,
            config.env_id,
            config.seed,
            num_envs=1,
            max_episode_steps=config.max_episode_steps,
        )
        save_frames = (
            config.visualize_rollouts and len(rollouts) < config.num_videos_save
        )
        rollout, success = single_rollout_fn(
            ts, rng_key, config, env, ic=ic, save_frames=save_frames
        )
        rollouts.append(rollout)
        successes.append(success)

    total_rollout_time = time.time() - rollout_start
    # import ipdb

    # ipdb.set_trace()

    rollouts = {k: [r[k] for r in rollouts] for k in rollouts[0]}
    metrics = {
        "avg_return": np.mean([np.sum(r) for r in rollouts["rewards"]]),
        "avg_success_rate": np.mean(successes),
        "num_successes": np.sum(successes),
        "avg_length": np.mean([len(r) for r in rollouts["rewards"]]),
        "total_rollout_time": total_rollout_time,
    }
    print(metrics)

    return rollouts, metrics


def single_rollout_fn(ts, rng_key, config: ConfigDict, env, ic=None, save_frames=False):
    obs, _ = env.reset(seed=config.seed)
    obs, _ = env.reset_to_state(ic)

    # TODO: maybe put this in a wrapper
    if config.env == "MAZE":
        if "meta" in obs:
            obs_input = np.concatenate(
                (obs["observation"], obs["desired_goal"], obs["meta"]), axis=-1
            )
        else:
            obs_input = np.concatenate((obs["observation"], obs["desired_goal"]))
    else:
        obs_input = obs

    rollout = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_observations": [],
        "frames": [],
    }

    t = 0
    success = False
    done = False

    while not done:
        # for ensemble
        if config.num_policies > 1:
            policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
            rng_key = policy_rng_keys[0]
            if config.policy_cls == "mlp":
                action = jax.vmap(
                    lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obs_input)
                )(ts.params, policy_rng_keys[1:])
                action = action.mean(axis=0)
            else:
                mean, logvar = jax.vmap(
                    lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obs_input)
                )(ts.params, policy_rng_keys[1:])
                action = mean.mean(axis=0)  # use mean for evaluation
        else:
            action = ts.apply_fn(ts.param, rng_key, obs=obs_input)

        action = action.squeeze()

        if isinstance(action, jnp.ndarray):
            # env does not work with jax array
            action = np.array(action.tolist())

        next_obs, reward, done, truncated, info = env.step(action)

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

        if info["success"]:
            success = True

        rollout["observations"].append(obs_input)
        rollout["actions"].append(action)
        rollout["rewards"].append(reward)
        rollout["dones"].append(done)
        rollout["next_observations"].append(next_obs_input)

        if save_frames:
            img = env.render()  # this returns a tuple

            if config.env == "MW":
                img = np.rot90(img, k=2)

            # H x W x C
            rollout["frames"].append(np.array(img))

        t += 1
        obs_input = next_obs_input
        obs = next_obs

        done = done or truncated

    return rollout, success


def async_rollout_fn(ts, rng_key, config: ConfigDict, env, save_frames: bool = False):
    rollout_start = time.time()

    obs, _ = env.reset(seed=config.seed)
    obs, _ = env.reset()

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
    successes = [False for _ in range(config.num_eval_envs)]
    dones = [False for _ in range(config.num_eval_envs)]
    lengths = [0 for _ in range(config.num_eval_envs)]

    rollouts = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_observations": [],
        "frames": [],
    }

    while not all(dones):
        # for ensemble
        if config.num_policies > 1:
            policy_rng_keys = jax.random.split(rng_key, config.num_policies + 1)
            rng_key = policy_rng_keys[0]
            if config.policy_cls == "mlp":
                action = jax.vmap(
                    lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obs_input)
                )(ts.params, policy_rng_keys[1:])
                action = action.mean(axis=0)
            else:
                mean, logvar = jax.vmap(
                    lambda param, rng_key: ts.apply_fn(param, rng_key, obs=obs_input)
                )(ts.params, policy_rng_keys[1:])
                # import ipdb

                # ipdb.set_trace()
                action = mean.mean(axis=0)  # use mean for evaluation
        else:
            action = ts.apply_fn(ts.param, rng_key, obs=obs_input)

        action = action.squeeze()

        if isinstance(action, jnp.ndarray):
            # env does not work with jax array
            action = np.array(action.tolist())

        # import ipdb

        # ipdb.set_trace()
        next_obs, reward, done, truncated, info = env.step(action)

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

        success = info["success"]

        if any(success):
            import ipdb

            ipdb.set_trace()

        successes = np.array(success) | np.array(successes)
        for i, s in enumerate(success):
            if not s:
                lengths[i] += 1

        dones = np.array(done) | np.array(dones) | np.array(truncated)

        rollouts["observations"].append(obs_input)
        rollouts["actions"].append(action)
        rollouts["rewards"].append(reward)
        rollouts["dones"].append(done)
        rollouts["next_observations"].append(next_obs_input)

        if save_frames:
            imgs = env.call("render")  # this returns a tuple

            if config.env == "MW":
                for indx, img in enumerate(imgs):
                    imgs[indx] = np.rot90(img, k=2)

            # N x H x W x C
            rollouts["frames"].append(np.array(imgs))

        t += 1
        obs_input = next_obs_input
        obs = next_obs

        # print("t: ", t, "reward: ", reward, "done: ", done, "truncated: ", truncated)
        # print("t: ", t, "reward: ", reward, "dones: ", dones, "successes: ", successes)

    # print("===" * 50)

    total_rollout_time = time.time() - rollout_start

    # T x N x D -> # N x T x D
    for k, v in rollouts.items():
        if len(v) > 0:
            rollouts[k] = np.swapaxes(np.array(rollouts[k]), 0, 1)

    # import ipdb

    # ipdb.set_trace()

    metrics = {
        "avg_return": np.mean(rollouts["rewards"].sum(axis=1)),
        "avg_success_rate": np.mean(successes),
        "num_successes": np.sum(successes),
        "avg_length": np.mean(lengths),
        "total_rollout_time": total_rollout_time,
    }

    print(metrics)

    if config.env == "D4RL":
        import gym

        env = gym.make(config.env_id).unwrapped
        metrics["avg_normalized_return"] = env.get_normalized_score(
            metrics["avg_return"]
        )

    return rollouts, metrics
