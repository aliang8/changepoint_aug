"""
Sample usage:
    python3 augment_bc_dataset.py \
        --env_name metaworld-assembly-v2 \
        --root_dir /scr/aliang80/changepoint_aug \
        --dataset_file datasets/expert_dataset/image_True/assembly-v2_100_noise_0.3 \
        --aug_type cp \
        --num_cps 50 \
        --num_augmentations_per_cp 1 \
        --num_expert_steps 20 \
        --num_pertubation_steps 0
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
@click.option("--image_based", default=True)
@click.option("--noise_std", default=0.1)
def main(
    env_name: str,
    root_dir: str,
    dataset_file: str,
    aug_type: str,
    num_expert_steps: int,
    num_pertubation_steps: int,
    num_cps: int = 100,
    num_augmentations_per_cp: int = 1,
    image_based: bool = True,
    noise_std: float = 0.1,
):
    # load base expert dataset
    print("load dataset from ", dataset_file)
    dataset_file = os.path.join(root_dir, dataset_file)
    expert_trajectories = serialize.load(dataset_file)
    print("number of expert trajectories: ", len(expert_trajectories))

    augmentations = []

    # create environment
    env = create_single_env(env_name, seed=521, image_based=False, noise_std=noise_std)

    # load expert policy
    expert_policy = SawyerAssemblyV2Policy()

    # get changepoint states
    # cp_timesteps = []
    # for trajectory in expert_trajectories:
    #     num_timesteps = len(trajectory.obs)
    #     found_cp = False
    #     for timestep in range(num_timesteps):
    #         # get the steps right before picking up the object
    #         if (
    #             trajectory.infos[timestep]["grasp_reward"] > 0.4
    #             and not trajectory.infos[timestep]["grasp_success"]
    #         ):
    #             found_cp = True
    #             break
    #         # if trajectory.infos[timestep]["in_place_reward"] > 0.4:
    #         #     found_cp = True
    #         #     break
    #     if found_cp:
    #         cp_timesteps.append(timestep)
    #     else:
    #         cp_timesteps.append(-1)
    cp_timesteps = [
        23,
        24,
        28,
        28,
        26,
        29,
        27,
        29,
        25,
        24,
        25,
        22,
        39,
        31,
        25,
        27,
        28,
        28,
        25,
        27,
        26,
        26,
        32,
        32,
        28,
        24,
        36,
        27,
        25,
        25,
        28,
        24,
        23,
        25,
        22,
        26,
        27,
        35,
        31,
        24,
        23,
        27,
        30,
        28,
        25,
        23,
        22,
        26,
        37,
        27,
        29,
        25,
        29,
        25,
        26,
        27,
        26,
        25,
        28,
        31,
        25,
        23,
        26,
        25,
        27,
        23,
        25,
        25,
        29,
        29,
        26,
        23,
        25,
        24,
        36,
        29,
        29,
        29,
        26,
        26,
        25,
        26,
        26,
        25,
        30,
        30,
        25,
        29,
        25,
        27,
        28,
        26,
        26,
        26,
        31,
        26,
        26,
        26,
        25,
        25,
    ]

    print(cp_timesteps)
    for traj_indx, cp_timestep in tqdm.tqdm(enumerate(cp_timesteps)):
        info = expert_trajectories[traj_indx].infos
        orig_state = info[cp_timestep]["state"]

        obs, _ = env.reset()
        # reset env to a randomly sampled state
        qpos = info[cp_timestep]["qpos"]
        qvel = info[cp_timestep]["qvel"]

        prev_obs = info[cp_timestep - 1]["state"]
        # print(prev_obs.shape)
        # also need to reset the task
        # first freeze the env
        env.unwrapped._freeze_rand_vec = True
        env.unwrapped._last_rand_vec = info[0]["last_rand_vec"]
        env.unwrapped._target_pos = info[0]["task"]
        env.reset_model()
        env.set_env_state((qpos, qvel))
        env.unwrapped._prev_obs = prev_obs

        # perturb by taking some random actions
        # for _ in range(num_pertubation_steps):
        #     rand_action = env.action_space.sample()
        #     # don't open the gripper
        #     rand_action[-1] = 0
        #     obs, _, _, _, _ = env.step(rand_action)

        # if hasattr(env, "unwrapped"):
        #     state_obs = env.unwrapped._get_obs()
        # else:
        #     state_obs = env._get_obs()

        obss = [obs]
        acts = []
        infos = []
        rews = []

        grasp_success = False
        obs = env.unwrapped._get_obs()

        # rollout expert for num_expert_steps
        for _ in range(num_expert_steps):
            # action, _ = expert_policy.predict(obs)
            # action = expert_policy.get_action(state_obs)
            action = expert_policy.get_action(obs)
            obs, rew, _, _, info = env.step(action)
            print(info["grasp_success"], info["grasp_reward"])

            # add env info
            if "metaworld" in env_name.lower():
                info["qpos"] = env.get_env_state()[0]
                info["qvel"] = env.get_env_state()[1]
                info["task"] = env.unwrapped._target_pos
                info["last_rand_vec"] = env.unwrapped._last_rand_vec

            obss.append(obs)
            acts.append(action)
            rews.append(rew)
            infos.append(info)

            # if hasattr(env, "unwrapped"):
            #     state_obs = env.unwrapped._get_obs()
            # else:
            #     state_obs = env._get_obs()

            if info["grasp_success"]:
                grasp_success = True

        print("grasp success: ", grasp_success)
        if grasp_success:
            # create augmentation
            augmentation = TrajectoryWithRew(
                acts=np.array(acts, dtype=np.float32),
                infos=infos,
                obs=np.array(obss, dtype=np.float32),
                rews=np.array(rews, dtype=np.float32),
                terminal=False,
            )
            augmentations.append(augmentation)

    # for _ in tqdm.tqdm(range(num_cps)):
    #     # sample random expert trajectory
    #     rollout_indx = np.random.randint(len(expert_trajectories))
    #     n_timesteps = len(expert_trajectories[rollout_indx].obs)
    #     if aug_type == "random":
    #         # sample random timestep as changepoint
    #         timestep = np.random.randint(1, n_timesteps - num_expert_steps)
    #     elif aug_type == "cp":
    #         # sample a changepoint timestep
    #         found_cp = False
    #         timestep = -1

    #         # heuristic-based changepoint detection
    #         for timestep in range(n_timesteps):
    #             if (
    #                 expert_trajectories[rollout_indx].infos[timestep]["in_place_reward"]
    #                 > 0.4
    #             ):
    #                 found_cp = True
    #                 break
    #     elif aug_type == "td":
    #         pass
    #     elif aug_type == "copycat":
    #         pass
    #     elif aug_type == "influence":
    #         pass

    #     info = expert_trajectories[rollout_indx].infos

    #     for _ in range(num_augmentations_per_cp):
    #         obs, _ = env.reset()
    #         # reset env to a randomly sampled state
    #         qpos = info[timestep]["qpos"]
    #         qvel = info[timestep]["qvel"]

    #         # also need to reset the task
    #         # first freeze the env
    #         env._freeze_rand_vec = True
    #         env._last_rand_vec = info[0]["last_rand_vec"]
    #         env._target_pos = info[0]["task"]
    #         env.reset_model()
    #         env.set_env_state((qpos, qvel))

    #         # perturb by taking some random actions
    #         for _ in range(num_pertubation_steps):
    #             rand_action = env.action_space.sample()
    #             # don't open the gripper
    #             rand_action[-1] = 0
    #             obs, _, _, _, _ = env.step(rand_action)

    #         obs = env._get_obs()
    #         obss = [obs]
    #         acts = []
    #         infos = []
    #         rews = []

    #         # rollout expert for num_expert_steps
    #         for _ in range(num_expert_steps):
    #             # action, _ = expert_policy.predict(obs)
    #             action = expert_policy.get_action(obs)
    #             obs, rew, _, _, info = env.step(action)

    #             # add env info
    #             if "metaworld" in env_name.lower():
    #                 info["qpos"] = env.get_env_state()[0]
    #                 info["qvel"] = env.get_env_state()[1]
    #                 info["task"] = env._target_pos
    #                 info["last_rand_vec"] = env._last_rand_vec

    #             obss.append(obs)
    #             acts.append(action)
    #             rews.append(rew)
    #             infos.append(info)

    #         # create augmentation
    #         augmentation = TrajectoryWithRew(
    #             acts=np.array(acts, dtype=np.float32),
    #             infos=infos,
    #             obs=np.array(obss, dtype=np.float32),
    #             rews=np.array(rews, dtype=np.float32),
    #             terminal=False,
    #         )
    #         augmentations.append(augmentation)

    output_dir = Path(root_dir) / dataset_file / f"aug_{aug_type}_{num_expert_steps}"
    print("saving to ", output_dir)
    output_dir.mkdir(exist_ok=True)
    serialize.save(output_dir, augmentations)


if __name__ == "__main__":
    main()
