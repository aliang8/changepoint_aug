"""
python3 bc.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 25 \
    --num_bc_epochs 100 \
    --mode train \
    --root_dir /scr/aliang80/changepoint_aug \
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

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args, **kwargs, net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])]
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
    augmentation_dataset_file: str = None,
):
    def _make_env():
        env = create_single_env(env_name, seed)
        env = RolloutInfoWrapper(env)
        return env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DummyVecEnv([_make_env for _ in range(n_eval_episodes)])
    evaluation_env = DummyVecEnv([_make_env for _ in range(n_eval_episodes)])
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

    print("number of transitions: ", len(transitions))

    policy = CustomPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )
    policy = policy.to(device)
    print(policy)
    print("policy parameters untrained: ", policy.action_net.weight[0][0])

    # create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
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
            log_rollouts_n_episodes=5,
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
        ckpt_path = (
            Path(root_dir)
            / "bc_policies"
            / env_name
            / f"demos_{num_demos}"
            / f"e_{num_bc_epochs}"
            / f"s_{seed}.zip"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving the trained policy to ", ckpt_path)
        save_policy(policy, ckpt_path)


if __name__ == "__main__":
    main()
