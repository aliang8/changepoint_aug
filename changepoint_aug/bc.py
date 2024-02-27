"""
DISPLAY=:3 CUDA_VISIBLE_DEVICES=5 python3 bc.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 100 \
    --num_bc_epochs 400 \
    --mode train \
    --n_eval_episodes 20 \
    --root_dir /scr/aliang80/changepoint_aug \
    --dataset_file datasets/expert_dataset/image_False/metaworld-assembly-v2_100_noise_0 \
    --seed 521 \
    --noise_std 0 \
    --image_based False \
    --log_interval 1000 \
    --log_to_wandb False \
    --augmentation_dataset_file  \
    
CUDA_VISIBLE_DEVICES=6 python3 bc.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 25 \
    --num_bc_epochs 400 \
    --mode train \
    --n_eval_episodes 10 \
    --root_dir /scr/aliang80/changepoint_aug \
    --dataset_file datasets/expert_dataset/image_True_pretrained_True_r3m_resnet50_add_proprio_stack_1/metaworld-assembly-v2_25_noise_0 \
    --seed 0 \
    --noise_std 0 \
    --image_based True \
    --log_interval 5000 \
    --use_pretrained_img_embeddings True \
    --embedding_name resnet50 \
    --pretrained_model r3m  \
    --history_window 1 \
    --log_to_wandb True
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
import wandb
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping
from metaworld.policies import *
from pathlib import *
from imitation.util.video_wrapper import VideoWrapper
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
from imitation.util.logger import configure

from stable_baselines3.sac.policies import Actor


# Custom MLP policy of three layers of size 128 each
class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(
            *args,
            **kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
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


def wandb_config():
    # Other users can overwrite this function to customize their wandb.init() call.
    wandb_tag = None  # User-specified tag for this run
    wandb_name_prefix = ""  # User-specified prefix for the run name
    wandb_kwargs = dict(
        project="changepoint_aug",
        monitor_gym=False,
        save_code=False,
    )  # Other kwargs to pass to wandb.init()
    wandb_additional_info = dict()

    return locals()


def wandb_init(
    config,
    env_name: str,
    seed: int,
    log_dir: str,
    wandb_name_prefix: str = "",
    wandb_tag: Optional[str] = "",
    wandb_kwargs: Mapping[str, Any] = {},
    wandb_additional_info: Mapping[str, Any] = {},
) -> None:
    """Putting everything together to get the W&B kwargs for wandb.init().

    Args:
        wandb_name_prefix: User-specified prefix for wandb run name.
        wandb_tag: User-specified tag for this run.
        wandb_kwargs: User-specified kwargs for wandb.init().
        wandb_additional_info: User-specific additional info to add to wandb experiment
            ``config``.
        log_dir: W&B logs will be stored in directory `{log_dir}/wandb/`.

    Raises:
        ModuleNotFoundError: wandb is not installed.
    """
    updated_wandb_kwargs: Mapping[str, Any] = {
        **wandb_kwargs,
        "name": f"{wandb_name_prefix}-{env_name}-seed{seed}",
        "tags": [env_name, f"seed{seed}"] + ([wandb_tag] if wandb_tag else []),
        "dir": log_dir,
    }
    wandb_config_dict = dict(**config)
    wandb_config_dict.update(wandb_additional_info)
    wandb.init(config=wandb_config_dict, **updated_wandb_kwargs)


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
@click.option("--n_frame_stack", default=3)
@click.option("--noise_std", default=0.0)
@click.option("--log_interval", default=500)  # number of batches between evals
@click.option("--log_to_wandb", default=True)
@click.option("--use_pretrained_img_embeddings", default=False)
@click.option("--embedding_name", default="resnet18")
@click.option("--pretrained_model", default="r3m")
@click.option("--history_window", default=1)
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
    n_frame_stack: int,
    noise_std: float,
    log_interval: int,
    log_to_wandb: bool,
    use_pretrained_img_embeddings: bool,
    embedding_name: str,
    pretrained_model: str,
    history_window: int,
    augmentation_dataset_file: str = None,
):
    # setup custom logger
    ckpt_dir = (
        Path(root_dir)
        / "bc_policies"
        / env_name
        / f"demos_{num_demos}"
        / f"e_{num_bc_epochs}"
        / f"image_{image_based}"
        / f"act_noise_{noise_std}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _make_env(seed):
        env = create_single_env(
            env_name,
            seed,
            image_based=image_based,
            noise_std=noise_std,
            freeze_rand_vec=True,
            use_pretrained_img_embeddings=use_pretrained_img_embeddings,
            add_proprio=True,
            embedding_name=embedding_name,
            history_window=history_window,
        )
        env = RolloutInfoWrapper(env)
        # env = VideoWrapper(env, single_video=False, directory=ckpt_dir / "videos")
        return env

    print("number of devices: ", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_env = DummyVecEnv(
        [
            partial(_make_env, seed=seed) for seed in range(n_eval_episodes)
        ]  # , start_method="fork"
    )
    if image_based and not use_pretrained_img_embeddings:
        # convert image observations to be CxHxW
        # env = sb3.common.vec_env.vec_transpose.VecTransposeImage(env)
        evaluation_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            evaluation_env
        )
        evaluation_env = sb3.common.vec_env.VecFrameStack(
            evaluation_env, n_stack=n_frame_stack
        )

    print(evaluation_env.observation_space)
    rng = np.random.default_rng(seed)

    print("load base expert dataset from ", dataset_file)
    dataset_file = Path(root_dir) / dataset_file
    expert_trajectories = serialize.load(dataset_file)
    print("number of expert trajectories in full dataset: ", len(expert_trajectories))

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

    if image_based and not use_pretrained_img_embeddings:
        policy_cls = CustomCNNPolicy
    else:
        policy_cls = CustomMLPPolicy
        # policy_cls = Actor

    policy = policy_cls(
        observation_space=evaluation_env.observation_space,
        action_space=evaluation_env.action_space,
        # net_arch=[128, 128, 128],
        # features_extractor=torch.nn.Flatten(),
        # features_dim=39
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )
    policy = policy.to(device)
    print(policy)

    print("policy parameters untrained: ", policy.action_net.weight[0][0])
    log_format_strs = ["stdout"]

    if log_to_wandb:
        log_format_strs.append("wandb")
        # run init
        config = wandb_config()
        config.update(
            {
                "env_name": env_name,
                "num_demos": num_demos,
                "seed": seed,
                "aug_type": aug_type,
                "noise_std": noise_std,
                "num_bc_epochs": num_bc_epochs,
                "n_eval_episodes": n_eval_episodes,
                "num_videos_save": num_videos_save,
                "image_based": image_based,
                "n_frame_stack": n_frame_stack,
                "log_interval": log_interval,
                "use_pretrained_img_embeddings": use_pretrained_img_embeddings,
                "embedding_name": embedding_name,
                "pretrained_model": pretrained_model,
            }
        )
        wandb_init(
            env_name=env_name,
            seed=seed,
            config=config,
            log_dir=ckpt_dir,
            wandb_name_prefix=f"bc-noise_std-{noise_std}-aug_type-{aug_type}-seed-{seed}-num_demos-{num_demos}-num_bc_epochs-{num_bc_epochs}-n_eval_episodes-{n_eval_episodes}-image_based-{image_based}-pretrained-{use_pretrained_img_embeddings}-embedding_name-{embedding_name}-pretrained_model-{pretrained_model}-add_proprio-True",
        )

    logger = configure(
        folder=ckpt_dir,
        format_strs=log_format_strs,
    )

    print(logger)

    # create BC trainer
    bc_trainer = bc.BC(
        observation_space=evaluation_env.observation_space,
        action_space=evaluation_env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=policy,
        device=device,
        custom_logger=logger,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        batch_size=32,
        l2_weight=0.01,
        ent_weight=0.01,
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
            log_interval=log_interval,  # number of batches between eval
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
        eval_file = ckpt_dir / f"eval_s_{seed}.txt"

        with open(eval_file, "w") as f:
            f.write(f"rew: {mean_reward}, std: {std_reward}, sr: {sr}")

        ckpt_path = ckpt_dir / f"s_{seed}.zip"
        print("Saving the trained policy to ", ckpt_path)
        save_policy(policy, ckpt_path)


if __name__ == "__main__":
    main()
