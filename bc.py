"""
CUDA_VISIBLE_DEVICES=5 python3 bc.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 25 \
    --num_bc_epochs 100 \
    --mode train \
    --n_eval_episodes 5 \
    --root_dir /scr/aliang80/changepoint_aug \
    --dataset_file datasets/expert_dataset/image_True/assembly-v2_100 \
    --seed 0 \
    --image_based True \
    --augmentation_dataset_file  \
"""

import os
import gym
import metaworld
from stable_baselines3 import PPO
import numpy as np
import click
import tqdm
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from metaworld.policies import *
from pathlib import *

from env_utils import create_single_env
from eval_policy import evaluate_policy
from datasets import concatenate_datasets
from imitation.util.util import save_policy
from imitation.algorithms.bc import reconstruct_policy
from imitation.data.huggingface_utils import TrajectoryDatasetSequence
from imitation.data import serialize
import imitation.data.rollout as rollout
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy


# Custom MLP policy of three layers of size 128 each
class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(
            *args, **kwargs, net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])]
        )


class CustomCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(
            *args,
            **kwargs,
            share_features_extractor=True,
            normalize_images=True,
            net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
        )


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--num_demos", default=25)
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--dataset_file", default="datasets/expert_dataset/assembly-v2_50")
@click.option("--num_bc_epochs", default=200)
@click.option("--n_eval_episodes", default=25)
@click.option("--num_videos_save", default=3)
@click.option("--resume", default=False)
@click.option("--mode", default="train")
@click.option("--aug_type", default="random")
@click.option("--seed", default=0)
@click.option("--image_based", default=False)
@click.option("--augmentation_dataset_file", default=None)
def main(
    env_name: str,
    num_demos: int,
    root_dir: str,
    dataset_file: str,
    num_bc_epochs: int,
    n_eval_episodes: int,
    num_videos_save: int,
    resume: bool,
    mode: str,
    aug_type: str,
    seed: int,
    image_based: bool,
    augmentation_dataset_file: str = None,
):
    def _make_env():
        env = create_single_env(env_name, seed, image_based=image_based)
        env = RolloutInfoWrapper(env)
        return env

    print("number of devices: ", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_env = SubprocVecEnv([_make_env for _ in range(n_eval_episodes)])
    if image_based:
        # convert image observations to be CxHxW
        # env = sb3.common.vec_env.vec_transpose.VecTransposeImage(env)
        evaluation_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            evaluation_env
        )

    print(evaluation_env.observation_space)
    rng = np.random.default_rng(seed)

    print("load base expert dataset from ", dataset_file)
    dataset_file = Path(root_dir) / dataset_file
    expert_trajectories = serialize.load(dataset_file)
    print("number of expert trajectories: ", len(expert_trajectories))

    if augmentation_dataset_file is not None:
        # make sure aug_type is in dataset files
        assert (
            aug_type in augmentation_dataset_file
        ), f"{aug_type} not in dataset file name"
        augmentation_trajectories = serialize.load(augmentation_dataset_file)
        print("number of augmentation trajectories: ", len(augmentation_trajectories))
        # combine augmentation traj with expert traj
        expert_dataset = expert_trajectories.dataset.select(np.arange(num_demos))
        combined_ds = concatenate_datasets(
            [expert_dataset, augmentation_trajectories.dataset]
        )
        combined_ds = TrajectoryDatasetSequence(combined_ds)
        transitions = rollout.flatten_trajectories(combined_ds)
    else:
        expert_trajectories = expert_trajectories[:num_demos]
        transitions = rollout.flatten_trajectories(expert_trajectories)

    print("obs shape: ", transitions[0]["obs"].shape)
    print("number of transitions: ", len(transitions))

    if image_based:
        policy_cls = CustomCNNPolicy
    else:
        policy_cls = CustomMLPPolicy

    policy = policy_cls(
        observation_space=evaluation_env.observation_space,
        action_space=evaluation_env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )
    policy = policy.to(device)
    print(policy)
    print("policy parameters untrained: ", policy.action_net.weight[0][0])

    # import ipdb

    # ipdb.set_trace()
    # create BC trainer
    bc_trainer = bc.BC(
        observation_space=evaluation_env.observation_space,
        action_space=evaluation_env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=policy,
        device=device,
    )

    # for debug
    print("policy parameters after ckpt: ", policy.action_net.weight[0][0])
    print(
        "bc policy parameters after ckpt: ",
        bc_trainer.policy.action_net.weight[0][0],
    )

    if mode == "train":
        print("Evaluating the untrained policy.")
        mean_reward, std_reward, sr = evaluate_policy(
            bc_trainer.policy,
            evaluation_env,
            n_eval_episodes=n_eval_episodes,
            render=False,
        )
        print(f"Reward before training: {mean_reward}, {std_reward}, {sr}")

        print("Training a policy using Behavior Cloning")
        bc_trainer.train(
            n_epochs=num_bc_epochs,
            progress_bar=True,
            log_interval=1000,  # number of batches between eval
            log_rollouts_n_episodes=n_eval_episodes,
            log_rollouts_venv=evaluation_env,
        )

    print("Evaluating the trained policy.")
    mean_reward, std_reward, sr = evaluate_policy(
        bc_trainer.policy,
        evaluation_env,
        n_eval_episodes=n_eval_episodes,
        render=False,
    )
    print(f"Reward after training: {mean_reward}, {std_reward}, {sr}")

    # save the trained policy
    if mode == "train":
        # write these eval stats to file
        ckpt_dir = (
            Path(root_dir)
            / "bc_policies"
            / env_name
            / f"demos_{num_demos}"
            / f"e_{num_bc_epochs}"
            / f"image_{image_based}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_file = ckpt_dir / f"eval_s_{seed}.txt"

        with open(eval_file, "w") as f:
            f.write(f"rew: {mean_reward}, std: {std_reward}, sr: {sr}")

        ckpt_path = ckpt_dir / f"s_{seed}.zip"
        print("Saving the trained policy to ", ckpt_path)
        save_policy(policy, ckpt_path)


if __name__ == "__main__":
    main()
