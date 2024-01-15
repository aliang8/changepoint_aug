from imitation.algorithms.bc import reconstruct_policy
from pathlib import *
import click
from bc import CustomPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from eval_policy import evaluate_policy
from env_utils import create_single_env
from imitation.data.wrappers import RolloutInfoWrapper


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--num_demos", default="25")
@click.option("--n_eval_episodes", default=25)
@click.option("--num_videos_save", default=3)
@click.option("--num_bc_epochs", default=100)
@click.option("--seed", default=0)
def main(
    env_name: str,
    num_demos: int,
    root_dir: str,
    n_eval_episodes: int,
    num_videos_save: int,
    num_bc_epochs: int = 100,
    seed: int = 0,
):
    def _make_env():
        env = create_single_env(env_name, seed)
        env = RolloutInfoWrapper(env)
        return env

    ckpt_path = (
        Path(root_dir)
        / "bc_policies"
        / env_name
        / f"demos_{num_demos}"
        / f"e_{num_bc_epochs}"
        / f"s_{seed}.zip"
    )

    policy = reconstruct_policy(policy_path=ckpt_path, device="cuda")

    evaluation_env = DummyVecEnv([_make_env for _ in range(n_eval_episodes)])

    # run evaluation
    mean_reward, std_reward, sr = evaluate_policy(
        policy,
        evaluation_env,
        n_eval_episodes=n_eval_episodes,
        render=False,
    )
    print(f"Reward after training: {mean_reward}, {std_reward}, {sr}")


if __name__ == "__main__":
    main()
