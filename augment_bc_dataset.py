"""
Sample usage:
    python3 augment_bc_dataset.py \
        --env_name metaworld-assembly-v2 \
        --root_dir /scr/aliang80/changepoint_aug \
        --dataset_file datasets/expert_dataset/assembly-v2_50 \
        --aug_type random \
        --num_cps 50 \
        --num_augmentations_per_cp 1 \
        --num_expert_steps 20 \
        --num_pertubation_steps 1
"""

import os
import numpy as np
import click
import tqdm
from pathlib import *
from imitation.data import serialize
from imitation.data.types import TrajectoryWithRew
from env_utils import create_single_env
from metaworld.policies import *


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--dataset_file", default="datasets/expert_dataset/assembly-v2_100")
@click.option("--aug_type", default="changepoint")
@click.option("--num_expert_steps", default=20)
@click.option("--num_pertubation_steps", default=1)
@click.option("--num_cps", default=100)
@click.option("--num_augmentations_per_cp", default=1)
def main(
    env_name: str,
    root_dir: str,
    dataset_file: str,
    aug_type: str,
    num_expert_steps: int,
    num_pertubation_steps: int,
    num_cps: int = 100,
    num_augmentations_per_cp: int = 1,
):
    # load base expert dataset
    print("load dataset from ", dataset_file)
    dataset_file = os.path.join(root_dir, dataset_file)
    expert_trajectories = serialize.load(dataset_file)
    print("number of expert trajectories: ", len(expert_trajectories))

    augmentations = []

    # create environment
    env = create_single_env(env_name)

    # load expert policy
    expert_policy = SawyerAssemblyV2Policy()

    for _ in tqdm.tqdm(range(num_cps)):
        # sample random expert trajectory
        rollout_indx = np.random.randint(len(expert_trajectories))
        n_timesteps = len(expert_trajectories[rollout_indx].obs)
        if aug_type == "random":
            # sample random timestep as changepoint
            timestep = np.random.randint(1, n_timesteps - num_expert_steps)
        elif aug_type == "cp":
            # sample a changepoint timestep
            found_cp = False
            timestep = -1
            for timestep in range(n_timesteps):
                if (
                    expert_trajectories[rollout_indx].infos[timestep]["in_place_reward"]
                    > 0.4
                ):
                    found_cp = True
                    break

        obs, _ = env.reset()
        for _ in range(num_augmentations_per_cp):
            # reset env to a randomly sampled state
            qpos = expert_trajectories[rollout_indx].infos[timestep]["qpos"]
            qvel = expert_trajectories[rollout_indx].infos[timestep]["qvel"]
            env.set_env_state((qpos, qvel))

            # perturb by taking some random actions
            for _ in range(num_pertubation_steps):
                obs, _, _, _, _ = env.step(env.action_space.sample())

            obss = [obs]
            acts = []
            infos = []
            rews = []

            # rollout expert for num_expert_steps
            for _ in range(num_expert_steps):
                # action, _ = expert_policy.predict(obs)
                action = expert_policy.get_action(obs)
                obs, rew, _, _, info = env.step(action)
                obss.append(obs)
                acts.append(action)
                rews.append(rew)
                infos.append(info)

            # create augmentation
            augmentation = TrajectoryWithRew(
                acts=np.array(acts, dtype=np.float32),
                infos=infos,
                obs=np.array(obss, dtype=np.float32),
                rews=np.array(rews, dtype=np.float32),
                terminal=False,
            )
            augmentations.append(augmentation)

    output_dir = (
        Path(root_dir)
        / "datasets"
        / "expert_dataset"
        / env_name
        / f"aug_{aug_type}_{num_expert_steps}"
    )
    serialize.save(output_dir, augmentations)


if __name__ == "__main__":
    main()
