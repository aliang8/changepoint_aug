"""
script to collect scripted demos for Metaworld
"""

import pickle
import random
import time
import tqdm
import numpy as np

import metaworld
from metaworld.policies import *
from env_utils import create_single_env
from imitation.data.types import TrajectoryWithRew
from imitation.data import serialize
from pathlib import *
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv


np.set_printoptions(suppress=True)

env_name = "metaworld-assembly-v2"
# env_name = "metaworld-drawer-open-v2"
num_episodes = 25
image_based = False
noise_std = 0
num_successful_traj = 0
traj_count = 0
use_pretrained_img_embeddings = True
embedding_name = "resnet50"
history_window = 1

trajectories = []

p = SawyerAssemblyV2Policy()
# p = SawyerDrawerOpenV2Policy()

while num_successful_traj < num_episodes:
    # create a new env with a different seed
    print(traj_count)
    env = create_single_env(
        env_name,
        seed=traj_count,
        image_based=image_based,
        noise_std=noise_std,
        freeze_rand_vec=True,
        use_pretrained_img_embeddings=use_pretrained_img_embeddings,
        add_proprio=True,
        embedding_name=embedding_name,
        history_window=history_window,
    )
    if image_based and not use_pretrained_img_embeddings:
        env = DummyVecEnv([lambda: env])
        env = sb3.common.vec_env.VecFrameStack(env, n_stack=3)
        obs = env.reset()
        info = {  # info
            "success": False,
            "near_object": 0.0,
            "grasp_success": False,
            "grasp_reward": 0.0,
            "in_place_reward": 0.0,
            "obj_to_target": 0.0,
            "unscaled_reward": 0.0,
        }
    else:
        obs, info = env.reset()

    if hasattr(env, "envs"):
        state_obs = env.envs[0].unwrapped._get_obs()
    elif hasattr(env, "unwrapped"):
        state_obs = env.unwrapped._get_obs()
    else:
        state_obs = env._get_obs()

    if image_based and not use_pretrained_img_embeddings:
        # make sure it is CxHxW
        obs = obs.squeeze(0)  # remove the env dimension
        obs = obs.transpose(2, 0, 1)

    timestep = 0
    done = False

    states = [obs]
    actions = []
    next_states = []
    rewards = []
    infos = []
    dones = []
    task_success = False

    # while timestep < 500 and not done:
    while timestep < 200:
        action = p.get_action(state_obs)
        if hasattr(env, "envs"):
            qpos, qvel = env.envs[0].get_env_state()
        else:
            qpos, qvel = env.get_env_state()

        if hasattr(env, "envs"):
            task = env.envs[0].unwrapped._target_pos
            last_rand_vec = env.envs[0].unwrapped._last_rand_vec
        elif hasattr(env, "unwrapped"):
            task = env.unwrapped._target_pos
            last_rand_vec = env.unwrapped._last_rand_vec
        else:
            task = env._target_pos
            last_rand_vec = env._last_rand_vec

        if "metaworld" in env_name.lower():
            info["qpos"] = qpos
            info["qvel"] = qvel
            info["task"] = task
            info["last_rand_vec"] = last_rand_vec
            info["state"] = state_obs
        infos.append(info)

        if image_based and not use_pretrained_img_embeddings:
            next_obs, reward, done, info = env.step([action])
            reward = reward.item()
            info = info[0]
        else:
            next_obs, reward, done, truncated, info = env.step(action)

        if int(info["success"]) == 1:
            # print(indx, " success")
            task_success = True
            done = True

        obs = next_obs
        if image_based and not use_pretrained_img_embeddings:
            # make sure it is CxHxW
            obs = obs.squeeze(0)  # remove the env dimension
            obs = obs.transpose(2, 0, 1)
            next_obs = next_obs.squeeze(0)  # remove the env dimension
            next_obs = next_obs.transpose(2, 0, 1)

        if hasattr(env, "envs"):
            state_obs = env.envs[0].unwrapped._get_obs()
        elif hasattr(env, "unwrapped"):
            state_obs = env.unwrapped._get_obs()
        else:
            state_obs = env._get_obs()

        timestep += 1

        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)
        rewards.append(reward)
        dones.append(done)

    if task_success:
        trajectory = dict(
            obs=np.array(states),
            acts=np.array(actions),
            rews=np.array(rewards),
            terminal=done,
            infos=np.array(infos),
        )
        # trajectory = TrajectoryWithRew(
        #     obs=np.array(states),
        #     acts=np.array(actions),
        #     rews=np.array(rewards),
        #     terminal=done,
        #     infos=np.array(infos),
        # )
        # print(trajectory.obs.shape)
        # print(trajectory.acts.shape)
        # print(trajectory.rews.shape)

        trajectories.append(trajectory)
        num_successful_traj += 1

    print(
        f"trajectory {traj_count}:",
        task_success,
        " total success: ",
        num_successful_traj,
    )
    traj_count += 1

print("done collecting trajectories")
# save trajectories
# output_path = Path(
#     f"datasets/expert_dataset/image_{image_based}_pretrained_{use_pretrained_img_embeddings}_r3m_{embedding_name}_add_proprio_stack_{history_window}"
# )
output_path = Path("/scr/aliang80/changepoint_aug/diffusion")
output_path.mkdir(parents=True, exist_ok=True)
pickle.dump(
    trajectories,
    open(output_path / f"{env_name}_{num_episodes}_noise_{noise_std}.pkl", "wb"),
)
# serialize.save(
#     output_path / f"{env_name}_{num_episodes}_noise_{noise_std}", trajectories
# )
print(output_path)
print(info)
