"""
script to collect scripted demos for Metaworld
"""
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

seed = 42
env_name = "metaworld-assembly-v2"
num_episodes = 100
image_based = True

env = create_single_env(env_name, seed, image_based=image_based)
p = SawyerAssemblyV2Policy()

trajectories = []
for indx in tqdm.tqdm(range(num_episodes)):
    obs, info = env.reset()

    if hasattr(env, "unwrapped"):
        state_obs = env.unwrapped._get_obs()
    else:
        state_obs = env._get_obs()

    if image_based:
        # make sure it is CxHxW
        assert len(obs.shape) == 3
        if obs.shape[-1] == 3:
            obs = obs.transpose(2, 0, 1)

    count = 0
    done = False

    states = [obs]
    actions = []
    next_states = []
    rewards = []
    infos = []
    dones = []

    # while count < 500 and not done:
    while count < 100:
        action = p.get_action(state_obs)
        next_obs, reward, _, truncated, info = env.step(action)
        if int(info["success"]) == 1:
            # print(indx, " success")
            done = True

        obs = next_obs
        if image_based:
            # make sure it is CxHxW
            assert len(obs.shape) == 3
            if obs.shape[-1] == 3:
                obs = obs.transpose(2, 0, 1)
                next_obs = next_obs.transpose(2, 0, 1)

        if hasattr(env, "unwrapped"):
            state_obs = env.unwrapped._get_obs()
        else:
            state_obs = env._get_obs()

        count += 1

        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)
        rewards.append(reward)
        dones.append(done)

        if "metaworld" in env_name.lower():
            info["qpos"] = env.get_env_state()[0]
            info["qvel"] = env.get_env_state()[1]
            if hasattr(env, "unwrapped"):
                info["task"] = env.unwrapped._target_pos
                info["last_rand_vec"] = env.unwrapped._last_rand_vec
            else:
                info["task"] = env._target_pos
                info["last_rand_vec"] = env._last_rand_vec
        infos.append(info)

    trajectory = TrajectoryWithRew(
        obs=np.array(states),
        acts=np.array(actions),
        rews=np.array(rewards),
        terminal=done,
        infos=np.array(infos),
    )

    trajectories.append(trajectory)

# save trajectories
output_path = Path(f"datasets/expert_dataset/image_{image_based}")
output_path.mkdir(parents=True, exist_ok=True)
serialize.save(output_path / f"assembly-v2_{num_episodes}", trajectories)

print(info)
