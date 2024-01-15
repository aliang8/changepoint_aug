"""
script to collect scripted demos for Metaworld
"""
import random
import time

import numpy as np

import metaworld
from metaworld.policies import *
from env_utils import create_single_env
from imitation.data.types import TrajectoryWithRew
from imitation.data import serialize
from pathlib import *

np.set_printoptions(suppress=True)

seed = 42
env_name = "metaworld-assembly-v2"
num_episodes = 50

env = create_single_env(env_name, seed)

p = SawyerAssemblyV2Policy()

trajectories = []
for indx in range(num_episodes):
    obs, info = env.reset()

    count = 0
    done = False

    states = [obs]
    actions = []
    next_states = []
    rewards = []
    infos = []
    dones = []

    # while count < 500 and not done:
    while count < 500:
        action = p.get_action(obs)
        next_obs, reward, _, truncated, info = env.step(action)
        if int(info["success"]) == 1:
            print(indx, " success")
            done = True

        obs = next_obs
        count += 1

        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)
        rewards.append(reward)
        dones.append(done)

        if "metaworld" in env_name.lower():
            info["qpos"] = env.get_env_state()[0]
            info["qvel"] = env.get_env_state()[1]
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
output_path = Path("datasets/expert_dataset")
output_path.mkdir(parents=True, exist_ok=True)
serialize.save(output_path / f"assembly-v2_{num_episodes}", trajectories)

print(info)
