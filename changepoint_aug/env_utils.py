import os
import tqdm
import cv2
import numpy as np
import imageio
import gymnasium as gym
import metaworld
from scipy import ndimage
from pathlib import *
import shutil
from changepoint_aug.wrappers import (
    PixelObservationWrapper,
    NoisyActionWrapper,
    PretrainedEmbeddingWrapper,
)
import stable_baselines3 as sb3


def make_video_from_trajectory(env_name: str, trajectory, video_dir: str):
    env = create_single_env(env_name, seed=0, image_based=False, freeze_rand_vec=True)

    obs, _ = env.reset()

    infos = trajectory.infos
    observations = trajectory.obs

    num_timesteps = len(infos)
    if len(observations.shape) == 4:
        # print(observations.shape)
        frames = observations.transpose(0, 2, 3, 1)
    else:
        frames = []

        for t in range(num_timesteps):
            info = infos[t]

            env.unwrapped._last_rand_vec = info["last_rand_vec"]
            env.unwrapped._target_pos = info["task"]
            if "assembly" in env_name.lower():
                # need to set the peg target location
                peg_pos = info["task"] - np.array([0.0, 0.0, 0.05])
                env.unwrapped.sim.model.body_pos[
                    env.unwrapped.model.body_name2id("peg")
                ] = peg_pos
                env.unwrapped.sim.model.site_pos[
                    env.unwrapped.model.site_name2id("pegTop")
                ] = info["task"]

            qpos = info["qpos"]
            qvel = info["qvel"]
            env.set_env_state((qpos, qvel))
            frame = env.unwrapped.render(offscreen=True)
            # flip this
            # frame = ndimage.rotate(frame, 180)
            frames.append(frame)

    episode_return = np.sum(trajectory.rews)
    success = trajectory.infos[-1]["success"]

    vid_path = os.path.join(
        video_dir,
        f"rollout_{round(episode_return, 2)}_{success}.mp4",
    )
    print(frames.shape)
    imageio.mimsave(vid_path, frames)


def create_single_env(
    env_name: str,
    seed: int = 0,
    image_based: bool = False,
    image_shape: tuple = (84, 84),
    noise_std: float = 0.1,
    freeze_rand_vec: bool = True,
    goal_observable: bool = True,
    use_pretrained_img_embeddings: bool = True,
    embedding_name: str = "resnet50",
    suite: str = "metaworld",
    history_window: int = 1,  # number of frame stack for pretrained image model
    add_proprio: bool = False,
) -> gym.Env:
    if "metaworld" in env_name.lower():
        env_name = env_name.replace("metaworld-", "")
        if goal_observable:
            env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
                env_name + "-goal-observable"
            ](seed=seed)
        else:
            env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
                env_name + "-goal-hidden"
            ](seed=seed)

        env.camera_name = "corner"  # corner2, corner3, behindGripper
        # to randomize it change the freeze rand vec
        # make sure seeded env has the same goal
        env._freeze_rand_vec = freeze_rand_vec
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.goal_space.seed(seed)

        # print(env.observation_space)

        # add wrapper such that the observation is an image
        if image_based:
            env = PixelObservationWrapper(env)

            if use_pretrained_img_embeddings:
                env = PretrainedEmbeddingWrapper(
                    env,
                    pretrained_model="r3m",
                    embedding_name=embedding_name,
                    suite=suite,
                    history_window=history_window,
                    add_proprio=add_proprio,
                )
            else:
                env = gym.wrappers.ResizeObservation(env, shape=image_shape)

        # noisy action wrapper
        env = NoisyActionWrapper(env, noise_std)

        # clip action wrapper
        env = gym.wrappers.ClipAction(env)
    else:
        raise NotImplementedError

    return env


def run_eval_rollouts(
    episode_length: int,
    num_episodes: int,
    env_name: str,
    policy,
    visualize: bool = False,
    video_dir: str = None,
    seed: int = 0,
    image_based: bool = False,
    noise_std: float = 0.1,
):
    # note, this is mainly for visualization, to generate eval use the
    # code from imitation library: https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html
    episode_returns = []

    env = create_single_env(
        env_name, seed, image_based=image_based, noise_std=noise_std
    )

    # first delete video_dir if it exists
    if video_dir is not None:
        video_dir = Path(video_dir)
        if video_dir.exists():
            shutil.rmtree(video_dir)

    video_dir.mkdir(parents=True, exist_ok=True)

    for rollout_idx in tqdm.tqdm(range(num_episodes)):
        frames = []
        states = None  # initial state
        obs, _ = env.reset()
        episode_return = 0

        success = False

        for t in range(episode_length):
            actions, states = policy.predict(
                obs,
                state=states,
                deterministic=True,
            )
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            if infos["success"]:
                success = True

            if visualize:
                frame = env.render()
                # flip this
                frame = ndimage.rotate(frame, 180)

                if "metaworld" in env_name.lower():
                    # this is for Metaworld, add some text for debugging
                    r = round(infos["in_place_reward"], 3)

                    frame = cv2.putText(
                        img=np.copy(frame),
                        text=f"{t}: {r}",
                        org=(400, 400),
                        fontFace=1,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=1,
                    )

                frames.append(frame)

            obs = next_obs
            episode_return += rewards

        episode_returns.append(episode_return)

        if video_dir is not None:
            vid_path = os.path.join(
                video_dir,
                f"rollout_{rollout_idx}_{round(episode_return, 2)}_{success}.mp4",
            )
            imageio.mimsave(vid_path, frames)
