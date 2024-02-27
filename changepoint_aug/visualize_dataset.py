"""
script to visualize some of the trajectories in the dataset

python3 visualize_dataset.py \
    --env_name metaworld-assembly-v2 \
    --root_dir /scr/aliang80/changepoint_aug \
    --dataset_file datasets/expert_dataset/image_True/metaworld-assembly-v2_100_noise_0 \
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
from env_utils import make_video_from_trajectory


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

    for indx, traj in enumerate(trajs):
        make_video_from_trajectory(env_name, traj, video_dir)


if __name__ == "__main__":
    main()
