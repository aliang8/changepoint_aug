"""
script for evaluating a trained BC policy

CUDA_VISIBLE_DEVICES=5 python3 test_bc_policy.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 100 \
    --num_bc_epochs 100 \
    --n_eval_episodes 5 \
    --num_videos_save 3 \
    --root_dir /scr/aliang80/changepoint_aug \
    --seed 0 \
    --image_based True
"""

from imitation.algorithms.bc import reconstruct_policy
from pathlib import *
import click
from bc import CustomMLPPolicy, CustomCNNPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from eval_policy import evaluate_policy
from env_utils import create_single_env
from imitation.data.wrappers import RolloutInfoWrapper
from env_utils import run_eval_rollouts


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--num_demos", default="25")
@click.option("--n_eval_episodes", default=25)
@click.option("--num_videos_save", default=3)
@click.option("--num_bc_epochs", default=100)
@click.option("--image_based", default=False)
@click.option("--seed", default=0)
def main(
    env_name: str,
    num_demos: int,
    root_dir: str,
    n_eval_episodes: int,
    num_videos_save: int,
    num_bc_epochs: int = 100,
    image_based: bool = False,
    seed: int = 0,
):
    def _make_env():
        env = create_single_env(env_name, seed, image_based=image_based)
        env = RolloutInfoWrapper(env)
        return env

    ckpt_dir = (
        Path(root_dir)
        / "bc_policies"
        / env_name
        / f"demos_{num_demos}"
        / f"e_{num_bc_epochs}"
        / f"image_{image_based}"
    )

    ckpt_path = ckpt_dir / f"s_{seed}.zip"
    policy = reconstruct_policy(policy_path=ckpt_path, device="cuda")

    evaluation_env = SubprocVecEnv([_make_env for _ in range(n_eval_episodes)])

    # run evaluation
    mean_reward, std_reward, sr = evaluate_policy(
        policy,
        evaluation_env,
        n_eval_episodes=n_eval_episodes,
        render=False,
    )
    print(f"Reward after training: {mean_reward}, {std_reward}, {sr}")

    # visualize some rollouts
    print("visualizing some trajectories")
    run_eval_rollouts(
        episode_length=500,
        num_episodes=num_videos_save,
        env_name=env_name,
        policy=policy,
        visualize=True,
        video_dir=ckpt_dir / "videos",
        seed=seed,
        image_based=image_based,
    )


if __name__ == "__main__":
    main()
