"""
Script to query states and augment with few steps of data augmentations
"""

from absl import app, logging
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from pathlib import Path
import pickle
import optax
import jax
import tqdm
import torch
import wandb
import io
import json
import time
import cv2
import imageio
from PIL import Image
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from functools import partial
import gymnasium as gym
import matplotlib.pyplot as plt
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import sys
import os
import haiku as hk
import jax.scipy as jsp

sys.path.append("/scr/aliang80/active_imitation_learning/old")
from sac_torch import Actor
from sac_torch_mw import Actor as ActorMW

from sac_eval import make_env as make_env_sac
from sac_eval_mw import make_env as make_env_sac_mw

from active_imitation_learning.utils.data import load_maze_dataset, make_blob_dataset
from active_imitation_learning.trainers.q_sarsa import create_ts as create_ts_q
from active_imitation_learning.trainers.cvae import create_ts as create_ts_cvae
from active_imitation_learning.trainers.bc import create_ts as create_ts_bc
from active_imitation_learning.visualize import estimate_density
from active_imitation_learning.utils.env_utils import make_env

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_EGL"] = "egl"


_CONFIG = config_flags.DEFINE_config_file("config")

shorthands = {
    "env": "eid",
    "num_expert_steps_aug": "nes",
    "num_augmentations_per_state": "nap",
    "num_perturb_steps": "nps",
    "selection_metric": "selm",
    "selection": "sel",
    "lamb": "lam",
    "reweight_with_density": "rwd",
    "total_num_states": "tns",
    "load_randomly_sampled_states": "rand",
}


def create_aug_data_dir(config):
    filename = "d"
    for k, v in config.keys_to_include.items():
        filename += f"_{shorthands[k]}-{getattr(config, k)}"
    return filename


def postprocess_density(densities, lamb=1.0):
    """
    Postprocess density estimates to be between 0 and 1
    """
    # densities represent logprobs
    densities = np.exp(densities)

    # rescale to be between 0 and 1
    densities = densities / np.max(densities)

    # either multiply or raise to some power
    densities = densities**lamb

    # should i also apply a smoothing kernel?
    densities = np.convolve(densities, np.ones(5) / 5, mode="same")
    return densities


def compute_jacobian(ts, x):
    # jax.debug.breakpoint()

    # make a output_dim x num_params Jacobian
    flat_params, unravel = jax.flatten_util.ravel_pytree(ts.params)
    jacobian = jax.jacobian(
        lambda flat_params: ts.apply_fn(unravel(flat_params), None, x)
    )(flat_params)
    # jacobian should be 2 x num_params
    return jacobian


def compute_influence_scores(jacs, residuals, lam):
    G = jax.vmap(lambda j: j.T @ j)(jacs)
    G = jnp.sum(G, axis=0)

    # TODO: maybe increase lambda

    # evals = jnp.linalg.eigvalsh(G)
    # # first, we fix the negative eigenvalues
    # lam = jnp.max(jnp.where(evals < 0, -evals, 0.0))
    # evals = jnp.where(evals < 0, 0.0, evals)

    # # next, we adjust the positive eigenvalues
    # tol = 1e-4
    # lam += jnp.max(jnp.where(evals < tol, tol - evals, 0.0))

    # jax.debug.breakpoint()
    H_inv = jsp.linalg.solve(
        G + lam * jnp.eye(G.shape[0]), jnp.eye(G.shape[0])  # , assume_a="pos"
    )
    # jax.debug.breakpoint()s

    gs = jax.vmap(lambda j, r: j.T @ r)(jacs, residuals)
    # jax.debug.breakpoint()

    scores = jax.vmap(lambda g: g.T @ H_inv @ g)(gs)
    return scores


class QueryStateSelector:
    def __init__(self, config: FrozenConfigDict, dataset, wandb_run=None):
        # print current directory
        print("current dir: ", os.getcwd())

        self.config = config
        self.wandb_run = wandb_run

        if self.config.load_randomly_sampled_states:
            # TODO: we don't have actions here
            self.obss = np.array(dataset)
            action_dim = 2  # for the maze enviornment
        else:
            self.obss = dataset["observations"]
            self.actions = dataset["actions"]
            self.dones = dataset["dones"]
            self.infos = dataset["infos"] if "infos" in dataset else None
            action_dim = self.actions.shape[-1]
            start_indices = np.where(self.dones)[0]
            start_indices += 1
            self.start_indices = np.insert(start_indices, 0, 0)
            logging.info(f"starting indices: {self.start_indices}")

        obs_shape = self.obss.shape
        logging.info(f"obs shape: {obs_shape}, action dim: {action_dim}")

        self.selection_metric = config.selection_metric

        assert self.selection_metric in [
            "q_variance",
            "policy_variance",
            "influence_function",
        ]

        # TODO: implement sort to get best checkpoint

        # load density model
        if self.config.reweight_with_density:
            density_model_ckpt_path = (
                Path(config.root_dir)
                / "ray_results"
                / config.density_exp_name
                / f"{config.density_exp_name}_{config.density_model_ckpt}"
                / config.ckpt_dir
                / f"epoch_{config.density_ckpt_step}.pkl"
            )
            logging.info(f"density model ckpt path: {density_model_ckpt_path}")
            print(f"density model ckpt path: {density_model_ckpt_path}")

            config.load_from_ckpt = str(density_model_ckpt_path)
            self.density_model_ckpt_path = density_model_ckpt_path
            self.density_ts = create_ts_cvae(config, 2, None)
        else:
            self.density_model_ckpt_path = ""
            self.density_ts = None

        # load metric for computing metric for measuring
        # quality of a state
        model_ckpt_path = (
            Path(config.root_dir)
            / "ray_results"
            / config.exp_name
            / config.model_ckpt
            / config.ckpt_dir
            / f"epoch_{config.ckpt_step}.pkl"
        )
        self.model_ckpt_path = model_ckpt_path

        model_config_path = (
            Path(config.root_dir)
            / "ray_results"
            / config.exp_name
            / config.model_ckpt
            / "params.json"
        )
        with open(model_config_path, "r") as f:
            model_cfg = json.load(f)
            model_cfg = ConfigDict(model_cfg)
        self.model_cfg = model_cfg
        self.rng_seq = hk.PRNGSequence(model_cfg.seed)

        if self.selection_metric == "q_variance":
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_q(config, obs_shape, action_dim, None)
        elif (
            self.selection_metric == "policy_variance"
            or self.selection_metric == "influence_function"
        ):
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_bc(model_cfg, obs_shape, action_dim, next(self.rng_seq))

        logging.info(f"model ckpt path: {model_ckpt_path}")
        print(f"model ckpt path: {model_ckpt_path}")

        if self.config.reweight_with_density:
            self.density_estimates = self.compute_density_estimates()
            self.density_estimates = postprocess_density(
                self.density_estimates, self.config.lamb
            )

        # compute metric
        self.metric = self.compute_metric()
        if self.config.reweight_with_density:
            self.reweighted_metric = self.metric * self.density_estimates

    def compute_density_estimates(self):
        """
        Compute density estimates for the entire dataset
        """
        logging.info(f"computing density estimates")
        density_estimates = []

        chunk_size = 1000
        rng_keys = jax.random.split(self.rng_key, self.obss.shape[0] + 1)
        self.rng_key = rng_keys[0]
        obss_chunked = np.array_split(self.obss, self.obss.shape[0] // chunk_size)
        rng_keys_chunked = np.array_split(
            rng_keys[1:], self.obss.shape[0] // chunk_size
        )

        density_estimates = []
        for indx, chunk_obs in enumerate(obss_chunked):
            density = jax.vmap(
                lambda obs, key: estimate_density(
                    self.density_ts,
                    key,
                    obs=obs[:2],  # xy location
                    goal=obs[4:6],  # goal
                    kl_div_weight=0.0,
                    num_posterior_samples=self.config.num_posterior_samples,
                )
            )(chunk_obs, rng_keys_chunked[indx])

            density_estimates.append(density)

        density_estimates = np.concatenate(density_estimates)
        logging.info(f"done computing density estimates, {density_estimates.shape}")

        return density_estimates

    def compute_metric(self):
        metric = None
        start = time.time()
        if self.selection_metric == "q_variance":
            logging.info(f"computing q variance")
            rng_keys = jax.random.split(self.rng_key, self.obss.shape[0] + 1)
            self.rng_key = rng_keys[0]
            q_rng_keys = rng_keys[1:]

            def apply_noise(params, rng_key, state, action):
                noise_samples = jax.random.uniform(
                    rng_key,
                    (self.config.num_samples_qvar, action.shape[-1]),
                    minval=-0.1,
                    maxval=0.1,
                )
                q = jax.vmap(
                    lambda noise: self.ts.apply_fn(params, state, action + noise)
                )(noise_samples)
                return q

            def apply_action(params, states, actions):
                q = jax.vmap(
                    lambda state, rng_key, action: apply_noise(
                        params, rng_key, state, action
                    )
                )(states, q_rng_keys, actions)
                return q

            q = apply_action(self.ts.params, self.obss, self.actions)
            # [T, noise_samples]
            q = jnp.squeeze(q, axis=-1)
            metric = jnp.var(q, axis=1)

        elif self.selection_metric == "policy_variance":
            logging.info(f"computing policy variance over ensemble of policies")
            rng_key = next(self.rng_seq)
            policy_rng_keys = jax.random.split(rng_key, self.config.num_policies)
            policy_output = jax.vmap(
                lambda param, rng_key: self.ts.apply_fn(param, rng_key, obs=self.obss)
            )(self.ts.params, policy_rng_keys)

            if self.model_cfg.policy_cls == "mlp":
                pass
            elif self.model_cfg.policy_cls == "gaussian":
                mean, logvar = policy_output
                ensemble_mean = mean.mean(axis=0)
                variance = jnp.exp(logvar)

                # compute variance between ensemble
                # average over ensemble
                # see: https://arxiv.org/pdf/1612.01474.pdf
                action_variance = (variance + mean**2).mean(axis=0) - ensemble_mean**2

            metric = action_variance.sum(axis=-1)
            # metric = jnp.var(action_preds, axis=0).sum(axis=-1)

        elif self.selection_metric == "influence_function":
            # this depends on what we want to compute influence of point wrt to

            # let's just use the first policy in ensemble
            params = jax.tree_map(lambda x: x[0], self.ts.params)
            self.ts = self.ts.replace(params=params)
            rng_key = next(self.rng_seq)
            policy_rng = jax.random.split(rng_key)
            action_preds = self.ts.apply_fn(self.ts.params, policy_rng, self.obss)
            residuals = optax.squared_error(self.actions, action_preds)

            # residuals = jnp.mean(residuals, axis=0)

            # compute jacobian of network for each datapoint
            # num points x output dim x num params

            # need to chunk this, cannot materialize all the jacobians at once
            chunk_size = 1000
            chunked_inputs = [
                self.obss[i : i + chunk_size]
                for i in range(0, len(self.obss), chunk_size)
            ]

            residuals_chunked = [
                residuals[i : i + chunk_size]
                for i in range(0, len(residuals), chunk_size)
            ]

            logging.info("computing jacobians")

            @jax.jit
            def func(carry, inputs):
                inp, residual = inputs
                jacs = jax.vmap(jax.jit(compute_jacobian), in_axes=(None, 0))(
                    self.ts, inp
                )
                metric = compute_influence_scores(
                    jacs, residual, self.config.inf_fn_lambda
                )
                return carry, metric

            _, metric = jax.lax.scan(
                func,
                init=[],
                xs=(jnp.array(chunked_inputs[:-1]), jnp.array(residuals_chunked[:-1])),
            )

            final_metric = jax.jit(func)(
                None, (jnp.array(chunked_inputs[-1]), jnp.array(residuals_chunked[-1]))
            )[1]
            metric = np.concatenate(metric, axis=0)
            metric = np.concatenate([metric, final_metric], axis=0)

            logging.info(
                f"done computing influence function, time taken: {time.time() - start}"
            )

        assert metric.shape[0] == self.obss.shape[0]
        logging.info(f"metric max: {np.max(metric)}, min: {np.min(metric)}")
        return metric

    def visualize_selected_states(self, selected_indices, env=None):
        logging.info("visualizing selected states")

        imgs = []
        metrics = self.metric[np.array(selected_indices)]

        for state in self.obss[selected_indices]:
            obs, _ = env.reset_to_state(state)
            img = env.render()
            imgs.append(img)

        # visualized selected states to augment
        imgs = np.array(imgs)

        logging.info(f"visualizing {imgs.shape[0]} images")
        n = 4
        num_imgs = n**2
        imgs_chunked = np.array_split(imgs, max(imgs.shape[0] // num_imgs, 1))
        metrics_chunked = np.array_split(metrics, max(metrics.shape[0] // num_imgs, 1))

        augment_data_dir = create_aug_data_dir(self.config)
        logging.info(f"augment data dir: {augment_data_dir}")
        vis_dir = (
            Path(self.config.data_dir)
            / "augment_datasets"
            / augment_data_dir
            / "images"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)

        for chunk_indx, chunk in tqdm.tqdm(
            enumerate(imgs_chunked), desc="saving images"
        ):
            # make a grid of 5 x 5 images
            fig, axs = plt.subplots(n, n, figsize=(20, 20))
            axs_flat = axs.flatten()

            for indx, img in enumerate(chunk):
                ax = axs_flat[indx]
                ax.axis("off")
                ax.imshow(chunk[indx])

                # put metric as title
                ax.set_title(
                    f"{self.selection_metric}: {metrics_chunked[chunk_indx][indx]:.2f}"
                )

            save_path = vis_dir / f"selected_states_{chunk_indx}.png"

            # remove space between figures
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.tight_layout()
            plt.savefig(save_path)

    def select_states_per_traj(self):
        """
        Handles computation of the metric and reranking states based on metric.
        """
        # iterate over each trajectory and reweigh the metric
        selected_indices_traj = []
        selected_indices_global = []
        selected_state_traj_indx = []

        # pick top k states per trajectory
        for traj_indx in tqdm.tqdm(
            range(len(self.start_indices) - 1), desc="per traj states"
        ):
            start = self.start_indices[traj_indx]
            end = self.start_indices[traj_indx + 1]

            if self.config.reweight_with_density:
                metric = self.reweighted_metric[start:end]
            else:
                metric = self.metric[start:end]

            # skip if metric is below threshold
            if (
                self.selection_metric == "influence_function"
                and np.max(metric) < self.config.metric_threshold
            ):
                continue

            selected_indx = np.argpartition(metric, -self.config.top_k)[
                -self.config.top_k :
            ][::-1]
            selected_indices_traj.extend(selected_indx)
            selected_indx += start
            selected_indices_global.extend(selected_indx)
            selected_state_traj_indx.extend([traj_indx] * selected_indx.shape[0])

        return selected_indices_traj, selected_indices_global, selected_state_traj_indx

    def select_states_global(self):
        """
        Handles computation of the metric and reranking states based on metric.
        """
        assert self.config.top_k == 1

        if self.config.reweight_with_density:
            metric = self.reweighted_metric
        else:
            metric = self.metric

        selected_indices_global = np.argpartition(
            metric, -self.config.total_num_states
        )[-self.config.total_num_states :][::-1]

        # figure out which traj each selected indx belongs to
        selected_state_traj_indx = []
        selected_indices_traj = []

        if not self.config.load_randomly_sampled_states:
            for indx in selected_indices_global:
                for traj_indx in range(len(self.start_indices) - 1):
                    start = self.start_indices[traj_indx]
                    end = self.start_indices[traj_indx + 1]
                    if start <= indx < end:
                        selected_state_traj_indx.append(traj_indx)
                        selected_indices_traj.append(indx - start)
                        break

        return selected_indices_traj, selected_indices_global, selected_state_traj_indx


def run_augmentation(config, wandb_run=None):
    config = ConfigDict(config)

    # load model config from pretrained heuristic model
    model_config_path = (
        Path(config.root_dir)
        / "ray_results"
        / config.exp_name
        / config.model_ckpt
        / "params.json"
    )
    with open(model_config_path, "r") as f:
        model_cfg = json.load(f)
        model_cfg = ConfigDict(model_cfg)

    logging.info(f"loading model config from: {model_config_path}")

    # set random seed
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)

    # STEP 1: Load base dataset and pass it into each run
    logging.info("Loading dataset")
    if config.env == "MAZE" or config.env == "MW":
        if config.load_randomly_sampled_states:
            dataset_f = Path(config.data_dir) / config.data_file
            with open(dataset_f, "rb") as f:
                dataset = pickle.load(f)
            logging.info(f"number of random states: {len(dataset)}")
            infos = None
            config.selection = "global"
        else:
            dataset = load_maze_dataset(model_cfg)
    else:
        raise NotImplementedError

    env = make_env(
        config.env,
        config.env_id,
        config.seed,
        max_episode_steps=config.max_episode_steps,
        freeze_rand_vec=False,
    )
    obs, _ = env.reset(seed=config.seed)
    # print(dataset[:][0][0])
    # obs, _ = env.reset_to_state(dataset[:][0][0].numpy())

    # STEP 2: Query states to augment
    # pick which states to query based on metric
    selector = QueryStateSelector(config, dataset, wandb_run=wandb_run)

    if config.selection == "per_traj":
        selected_indices_traj, selected_indices_global, selected_state_traj_indx = (
            selector.select_states_per_traj()
        )
    elif config.selection == "global":
        selected_indices_traj, selected_indices_global, selected_state_traj_indx = (
            selector.select_states_global()
        )

    logging.info(f"number of selected states: {len(selected_indices_global)}")
    logging.info(f"selected indices: {selected_indices_global}")
    logging.info(f"selected indices traj: {selected_indices_traj}")
    logging.info(f"selected indices traj indx: {selected_state_traj_indx}")

    if config.visualize:
        selector.visualize_selected_states(selected_indices_global, env)

    # STEP 3: Augment selected states with a few steps of expert actions
    # Perturb states a little bit

    # load expert model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        config.oracle_model_ckpt_path,
        map_location=device,
    )
    envs = gym.vector.SyncVectorEnv([make_env_sac(config.env_id, 0, 0, False, "")])
    actor = Actor(envs).to(device)
    actor.load_state_dict(state_dict["actor"])
    actor.eval()

    rollouts = []

    selected_states = selector.obss[selected_indices_global]
    selected_infos = None
    if not config.load_randomly_sampled_states:
        if selector.infos is not None:
            selected_infos = [selector.infos[ind] for ind in selected_indices_global]

    # import ipdb

    # ipdb.set_trace()

    # save images and generate a video of the augmentation dataset
    imgs = []

    # total number of new transitions = (# augmentations per state) x (# num total trajs) x (# expert steps)
    for indx, state in tqdm.tqdm(
        enumerate(selected_states), desc="collecting augmentations"
    ):

        for _ in range(config.num_augmentations_per_state):
            if config.env == "MAZE":
                obs, _ = env.reset_to_state(state)
            elif config.env == "MW":
                # need to reset the env to the same seed, and then take the same actions to reset to
                # the selected state. not sure why metaworld doesn't have a better way of resetting to a desired state
                info = selected_infos[indx]
                seed = info["seed"]
                traj_indx = selected_state_traj_indx[indx]
                start = selector.start_indices[traj_indx]

                env = make_env(
                    config.env,
                    config.env_id,
                    seed,
                    max_episode_steps=config.max_episode_steps,
                    freeze_rand_vec=False,
                )
                obs, _ = env.reset()
                actions = selector.actions[start : start + selected_indices_traj[indx]]

                # take the sequence of actions to reach the state
                for action in actions:
                    env.step(action)

                # env_state = selected_infos[indx]["env_state"]
                # env.set_env_state(env_state)
                obs = env.get_obs()

            # take some perturbation steps
            for _ in range(config.num_perturb_steps):
                action = env.action_space.sample()
                next_obs, reward, done, truncated, info = env.step(action)
                obs = next_obs

            if config.visualize:
                img = np.array(env.render())
                img = cv2.putText(
                    img,
                    str(indx),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                imgs.append(img)

            for _ in range(config.num_expert_steps_aug):
                if config.env == "MAZE":
                    obs_input = np.concatenate(
                        [obs["observation"], obs["desired_goal"], obs["meta"]],
                    )
                else:
                    obs_input = obs

                if len(obs_input.shape) == 1:
                    obs_input = obs_input[None]

                # don't use the sampled action
                action_sample, _, action_mean = actor.get_action(
                    torch.Tensor(obs_input).to(device)
                )
                action = action_mean[0].detach().cpu().numpy()
                next_obs, reward, done, truncated, info = env.step(action)

                if config.visualize:
                    img = env.render()
                    imgs.append(img)

                if config.env == "MW":
                    info["env_state"] = env.get_env_state()

                    if len(obs.shape) == 1:
                        obs = obs[None]  # add extra dimension

                    if len(next_obs.shape) == 1:
                        next_obs = next_obs[None]

                rollouts.append((obs, next_obs, action, reward, done, truncated, info))

                # if info["success"]:
                #     break

                obs = next_obs

    logging.info(f"number of new transitions: {len(rollouts)}")

    # STEP 4: Save augmented dataset
    augment_data_dir = create_aug_data_dir(config)
    save_dir = Path(config.data_dir) / "augment_datasets" / augment_data_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / "dataset.pkl"
    logging.info(f"Saving augmented dataset to {save_file}")
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "rollouts": rollouts,
                "metadata": {
                    "density_model_ckpt_path": selector.density_model_ckpt_path,
                    "model_ckpt_path": selector.model_ckpt_path,
                    "num_transitions": len(rollouts),
                    "config": config.to_dict(),
                },
            },
            f,
        )

    if config.visualize:
        video_file = (
            Path(config.data_dir) / "augment_datasets" / augment_data_dir / "trajs.mp4"
        )
        imageio.mimsave(
            video_file,
            imgs,
            fps=10,
        )

    # finish and close the environments
    env.close()


# run with ray tune
param_space = {
    # "seed": tune.grid_search([0, 1, 2]),
    # "num_trajs": tune.grid_search([5, 10, 25, 50, 75, 100, 125, 150, 175, 200]),
    "num_trajs": tune.grid_search([5, 10]),
}


def main(_):
    config = _CONFIG.value

    if not config.smoke_test:
        # log the visualizations to wandb
        wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            entity="glamor",
            project="data_augmentation",
            name=config.exp_name,
            notes=config.notes,
            tags=config.tags,
            group=config.group_name if config.group_name else None,
            # track hyperparameters and run metadata
            config=config,
        )

        run_augmentation_fn = tune.with_resources(
            run_augmentation, {"cpu": 3, "gpu": 0.1}
        )

        config = config.to_dict()  # needs to be a dict to go into tune
        run_config = RunConfig(
            name="data_augmentation",
            local_dir="/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/ray_results",
            storage_path="/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/ray_results",
            log_to_file=False,
        )
        config.update(param_space)
        tuner = tune.Tuner(
            tune.with_parameters(run_augmentation_fn, wandb_run=wandb_run),
            param_space=config,
            run_config=run_config,
            # tune_config=tune.TuneConfig(
            #     trial_name_creator=trial_str_creator,
            #     trial_dirname_creator=trial_str_creator,
            # ),
        )
        results = tuner.fit()
        print(results)
    else:
        run_augmentation(config)


if __name__ == "__main__":
    app.run(main)
