import os
import tqdm
import cv2
import numpy as np
import imageio
import gymnasium as gym
import metaworld


class PixelObservationWrapper(gym.Wrapper):
    """Pixel observation wrapper for obtaining pixel observations.

    Instead of returning the default environment observation, the wrapped
    environment's render function is used to produce RGB pixel observations.

    This behaves like gym.wrappers.PixelObservationWrapper but returns a
    gym.spaces.Box observation space and observation instead of
    a gym.spaces.Dict.

    Args:
        env (gym.Env): The environment to wrap. This environment must produce
            non-pixel observations and have a Box observation space.
        headless (bool): If true, this creates a window to init GLFW. Set to
            true if running on a headless machine or with a dummy X server,
            false otherwise.

    """

    def __init__(self, env, headless=True):
        # if headless:
        #     # pylint: disable=import-outside-toplevel
        #     # this import fails without a valid mujoco license
        #     # so keep this here to avoid unecessarily requiring
        #     # a mujoco license everytime the wrappers package is
        #     # accessed.
        #     from mujoco_py import GlfwContext

        #     GlfwContext(offscreen=True)
        env.reset()
        # print(env.observation_space)
        env = gym.wrappers.PixelObservationWrapper(env, pixels_only=False)
        # print(env.observation_space)
        super().__init__(env)
        self._observation_space = env.observation_space["pixels"]

    @property
    def observation_space(self):
        """gym.spaces.Box: Environment observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
        """
        obs, info = self.env.reset(**kwargs)
        info["state"] = obs["state"]
        return obs["pixels"], info

    def step(self, action):
        """gym.Env step function.

        Performs one action step in the enviornment.

        Args:
            action (np.ndarray): Action of shape :math:`(A*, )`
                to pass to the environment.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
            float : Amount of reward returned after previous action.
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results.
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning).
        """
        obs, reward, done, truncated, info = self.env.step(action)
        info["state"] = obs["state"]
        return obs["pixels"], reward, done, truncated, info


def create_single_env(
    env_name: str,
    seed: int = 0,
    image_based: bool = False,
    image_shape: tuple = (64, 64),
) -> gym.Env:
    if "metaworld" in env_name.lower():
        env_name = env_name.replace("metaworld-", "")
        env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            env_name + "-goal-observable"
        ](seed=seed, render_mode="rgb_array")

        # different initialization every reset
        env.camera_name = "corner"  # corner2, corner3, behindGripper
        env._freeze_rand_vec = False
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.goal_space.seed(seed)

        # print(env.observation_space)

        # add wrapper such that the observation is an image
        if image_based:
            env = PixelObservationWrapper(env)
            # print(env.observation_space)
            env = gym.wrappers.ResizeObservation(env, shape=image_shape)
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
