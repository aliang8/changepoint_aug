import os
import tqdm
import cv2
import numpy as np
import imageio
import gymnasium as gym
import metaworld


def create_single_env(env_name: str, seed: int = 0) -> gym.Env:
    if "metaworld" in env_name.lower():
        env_name = env_name.replace("metaworld-", "")
        env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            env_name + "-goal-observable"
        ](seed=seed, render_mode="rgb_array")
    else:
        raise NotImplementedError

    # different initialization every reset
    env.camera_name = "corner"  # corner2, corner3, behindGripper
    env._freeze_rand_vec = False
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.goal_space.seed(seed)
    return env


def run_eval_rollouts(
    episode_length: int,
    num_episodes: int,
    env_name: str,
    policy,
    visualize: bool = False,
    video_dir: str = None,
):
    # note, this is mainly for visualization, to generate eval use the
    # code from imitation library: https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html
    episode_returns = []

    env = create_single_env(env_name)

    for rollout_idx in tqdm.tqdm(range(num_episodes)):
        frames = []
        states = None  # initial state
        obs, _ = env.reset()
        episode_return = 0

        for t in range(episode_length):
            actions, states = policy.predict(
                obs,
                state=states,
                deterministic=True,
            )
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            if visualize:
                frame = env.render()

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
            vid_path = os.path.join(video_dir, f"rollout_{rollout_idx}.mp4")
            imageio.mimsave(vid_path, frames)
