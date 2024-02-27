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
import matplotlib.pyplot as plt

import warnings
from typing import Any, Dict, Generic, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np
from gymnasium import spaces
import torch
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)

TObs = TypeVar("TObs", np.ndarray, Dict[str, np.ndarray])
from PIL import Image
from omegaconf import OmegaConf
import hydra
from r3m import load_r3m
import torchvision.transforms as T


def load_pretrained_model(
    pretrained_model: str,
    model_path: str = "",
    embedding_name: str = "",
    input_type=np.ndarray,
    *args,
    **kwargs,
):
    """
    Load the pretrained model based on the config corresponding to the embedding_name
    """

    if pretrained_model == "vc-1":
        config_path = os.path.join(model_path, "conf/model", embedding_name + ".yaml")
        print("Loading config path: %s" % config_path)
        config = OmegaConf.load(config_path)
        model, embedding_dim, transforms, metadata = hydra.utils.call(config)
    elif pretrained_model == "r3m":
        model = load_r3m(embedding_name)
        if embedding_name == "resnet18":
            embedding_dim = 512
        else:
            embedding_dim = 2048
        ## DEFINE PREPROCESSING
        transforms = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
        )  # ToTensor() divides by 255
        metadata = None

    model = model.eval()  # model loading API is unreliable, call eval to be double sure

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
        else:
            return transforms

    return model, embedding_dim, final_transforms(transforms), metadata


def get_proprioception(env: gym.Env, suite: str) -> Union[np.ndarray, None]:
    assert isinstance(env, gym.Env)
    if suite == "metaworld":
        return env.unwrapped._get_obs()[:4]
    elif suite == "adroit":
        # In adroit, in-hand tasks like pen lock the base of the hand
        # while other tasks like relocate allow for movement of hand base
        # as if attached to an arm
        if env.unwrapped.spec.id == "pen-v0":
            return env.unwrapped.get_obs()[:24]
        elif env.unwrapped.spec.id == "relocate-v0":
            return env.unwrapped.get_obs()[:30]
        else:
            print("Unsupported environment. Proprioception is defaulting to None.")
            return None
    elif suite == "dmc":
        # no proprioception used for dm-control
        return None
    else:
        print("Unsupported environment. Proprioception is defaulting to None.")
        return None


class PretrainedEmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a frozen vision model over the image observation.

    Args:
        env (Gym environment): the original environment
        suite (str): category of environment ["dmc", "adroit", "metaworld"]
        embedding_name (str): name of the embedding to use (name of config)
        history_window (int, 1) : timesteps of observation embedding to incorporate into observation (state)
        embedding_fusion (callable, 'None'): function for fusing the embeddings into a state.
            Defaults to concatenation if not specified
        obs_dim (int, 'None') : dimensionality of observation space. Inferred if not specified.
            Required if function != None. Defaults to history_window * embedding_dim
        add_proprio (bool, 'False') : flag to specify if proprioception should be appended to observation
        device (str, 'cuda'): where to allocate the model.
    """

    def __init__(
        self,
        env,
        embedding_name: str,
        suite: str,
        pretrained_model: str = "r3m",
        history_window: int = 1,
        fuse_embeddings: callable = None,
        obs_dim: int = None,
        device: str = "cuda",
        seed: int = None,
        add_proprio: bool = False,
        *args,
        **kwargs,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.embedding_buffer = (
            []
        )  # buffer to store raw embeddings of the image observation
        self.obs_buffer = []  # temp variable, delete this line later
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        if device == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device("cuda")
        else:
            print("Not using CUDA.")
            device = torch.device("cpu")
        self.device = device

        # get the embedding model
        embedding, embedding_dim, transforms, metadata = load_pretrained_model(
            pretrained_model=pretrained_model, embedding_name=embedding_name, seed=seed
        )
        embedding.to(device=self.device)
        # freeze the PVR
        for p in embedding.parameters():
            p.requires_grad = False
        self.embedding, self.embedding_dim, self.transforms = (
            embedding,
            embedding_dim,
            transforms,
        )

        # proprioception
        if add_proprio:
            self.get_proprio = lambda: get_proprioception(self, suite)
            proprio = self.get_proprio()
            self.proprio_dim = 0 if proprio is None else proprio.shape[0]
        else:
            self.proprio_dim = 0
            self.get_proprio = None

        # final observation space
        obs_dim = (
            obs_dim
            if obs_dim != None
            else int(self.history_window * self.embedding_dim + self.proprio_dim)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,)
        )

    def observation(self, observation):
        # i think we need to flip the image first
        observation = observation[::-1, :, :]

        # observation shape : (H, W, 3)
        inp = self.transforms(
            observation
        )  # numpy to PIL to torch.Tensor. Final dimension: (1, 3, H, W)
        inp = inp.to(self.device)
        with torch.no_grad():
            emb = (
                self.embedding(inp)
                .view(-1, self.embedding_dim)
                .to("cpu")
                .numpy()
                .squeeze()
            )
        # update observation buffer
        if len(self.embedding_buffer) < self.history_window:
            # initialization
            self.embedding_buffer = [emb.copy()] * self.history_window
        else:
            # fixed size buffer, replace oldest entry
            for i in range(self.history_window - 1):
                self.embedding_buffer[i] = self.embedding_buffer[i + 1].copy()
            self.embedding_buffer[-1] = emb.copy()

        # fuse embeddings to obtain observation
        if self.fuse_embeddings != None:
            obs = self.fuse_embeddings(self.embedding_buffer)
        else:
            # print("Fuse embedding function not given. Defaulting to concat.")
            obs = np.array(self.embedding_buffer).ravel()

        # add proprioception if necessary
        if self.proprio_dim > 0:
            proprio = self.get_proprio()
            obs = np.concatenate([obs, proprio])
        return obs

    def get_obs(self):
        return self.observation(self.env.observation(None))

    def get_image(self):
        return self.env.get_image()

    def reset(self, **kwargs):
        self.embedding_buffer = []  # reset to empty buffer
        return super().reset(**kwargs)


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


class NoisyActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        noise_std: float = 0.1,
    ):
        """Initializes the :class:`NoisyActionWrapper` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
        """
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"

        gym.ActionWrapper.__init__(self, env)
        self.action_space = env.action_space
        self.noise_std = noise_std

    def action(self, action):
        """
        Args:
            action: The action to rescale

        Returns:
            noisy action
        """
        # print("old action: ", action)
        action = action + np.random.normal(0, self.noise_std, size=action.shape)
        # print("new action: ", action)
        return action
