"""
script for evaluating a trained BC policy

CUDA_VISIBLE_DEVICES=5 python3 test_bc_policy.py \
    --env_name metaworld-assembly-v2 \
    --num_demos 100 \
    --num_bc_epochs 400 \
    --n_eval_episodes 2 \
    --num_videos_save 3 \
    --root_dir /scr/aliang80/changepoint_aug \
    --image_based True \
    --noise_std 0.0 \
    --seed 521 \
    --use_pretrained_img_embeddings True \
    --embedding_name resnet50 \
    --pretrained_model r3m \
    --save_video True
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
from functools import partial
from imitation.util.video_wrapper import VideoWrapper
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--num_demos", default=25)
@click.option("--n_eval_episodes", default=25)
@click.option("--num_videos_save", default=3)
@click.option("--num_bc_epochs", default=100)
@click.option("--image_based", default=False)
@click.option("--noise_std", default=0.0)
@click.option("--seed", default=0)
@click.option("--save_video", default=False)
@click.option("--use_pretrained_img_embeddings", default=False)
@click.option("--embedding_name", default="resnet18")
@click.option("--pretrained_model", default="r3m")
def main(
    env_name: str,
    num_demos: int,
    root_dir: str,
    n_eval_episodes: int,
    num_videos_save: int,
    num_bc_epochs: int = 100,
    image_based: bool = False,
    noise_std: float = 0.1,
    seed: int = 0,
    save_video: bool = False,
    use_pretrained_img_embeddings: bool = False,
    embedding_name: str = "",
    pretrained_model: str = "",
):
    ckpt_dir = (
        Path(root_dir)
        / "bc_policies"
        / env_name
        / f"demos_{num_demos}"
        / f"e_{num_bc_epochs}"
        / f"image_{image_based}"
        / f"act_noise_{noise_std}"
    )

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
        )
        env = RolloutInfoWrapper(env)
        return env

    ckpt_path = ckpt_dir / f"s_{seed}.zip"
    print("loading policy from", ckpt_path)
    policy = reconstruct_policy(policy_path=ckpt_path, device="cuda")

    evaluation_env = DummyVecEnv(
        [partial(_make_env, seed=seed + 6) for seed in range(n_eval_episodes)]
    )

    if save_video:
        evaluation_env = VecVideoRecorder(
            evaluation_env,
            record_video_trigger=lambda x: x == 0,
            video_folder=ckpt_dir / "videos",
        )

    # run evaluation
    mean_reward, std_reward, sr = evaluate_policy(
        policy,
        evaluation_env,
        n_eval_episodes=n_eval_episodes,
        render=False,
        deterministic=True,
    )
    print(f"Reward after training: {mean_reward}, {std_reward}, {sr}")

    evaluation_env.close()
    # visualize some rollouts
    # print("visualizing some trajectories")
    # run_eval_rollouts(
    #     episode_length=150,
    #     num_episodes=num_videos_save,
    #     env_name=env_name,
    #     policy=policy,
    #     visualize=True,
    #     video_dir=ckpt_dir / f"videos_s_{seed}",
    #     seed=seed,
    #     image_based=image_based,
    #     noise_std=noise_std,
    # )


if __name__ == "__main__":
    main()
