"""
script to visualize some of the trajectories in the dataset

python3 visualize_dataset.py \
    --env_name metaworld-assembly-v2 \
    --root_dir /scr/aliang80/changepoint_aug \
    --dataset_file datasets/expert_dataset/assembly-v2_50 \
    --num_videos_save 2
"""

import os
import click
from pathlib import *
from imitation.data import serialize
import numpy as np
import tqdm
import imageio
from scipy import ndimage
import matplotlib.pyplot as plt
from env_utils import create_single_env


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--dataset_file", default="datasets/expert_dataset/assembly-v2_100")
@click.option("--num_videos_save", default=2)
def main(
    env_name: str,
    root_dir: str,
    dataset_file: str,
    num_videos_save: int,
):
    data_path = Path(root_dir) / dataset_file
    expert_trajectories = serialize.load(data_path)
    print("number of expert trajectories: ", len(expert_trajectories))

    # sample random trajectories
    indices = np.random.choice(len(expert_trajectories), num_videos_save, replace=False)
    trajs = [expert_trajectories[int(i)] for i in indices]

    video_dir = data_path / "videos"
    video_dir.mkdir(exist_ok=True)

    if "images" not in trajs[0].infos:
        # create environment, need to reset to state
        env = create_single_env(env_name)

        for indx, traj in enumerate(trajs):
            env.reset()
            infos = traj.infos

            env._freeze_rand_vec = True
            env._last_rand_vec = infos[0]["last_rand_vec"]
            env._target_pos = infos[0]["task"]
            env.reset_model()

            images = []

            print(indx, len(infos))
            for ts in range(len(infos)):
                qpos = infos[ts]["qpos"]
                qvel = infos[ts]["qvel"]
                env.set_env_state((qpos, qvel))
                img = env.render()
                img = ndimage.rotate(img, 180)
                images.append(img)

            # save images to file
            imageio.mimsave(video_dir / f"rollout_{indx}.mp4", images)


if __name__ == "__main__":
    main()
