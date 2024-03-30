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
import jax.scipy as jsp

sys.path.append("/scr/aliang80/changepoint_aug/changepoint_aug/old")
from sac_torch import Actor
from sac_torch_mw import Actor as ActorMW

from sac_eval import make_env as make_env_sac
from sac_eval_mw import make_env as make_env_sac_mw

from changepoint_aug.density_estimation.data import load_pkl_dataset, make_blob_dataset
from changepoint_aug.density_estimation.trainers.q_sarsa import create_ts as create_ts_q
from changepoint_aug.density_estimation.trainers.cvae import create_ts as create_ts_cvae
from changepoint_aug.density_estimation.trainers.bc import create_ts as create_ts_bc
from changepoint_aug.density_estimation.visualize import estimate_density
from changepoint_aug.density_estimation.utils import make_env

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
}


def create_aug_filename(config):
    filename = "augment_dataset"
    for k, v in shorthands.items():
        if hasattr(config, k):
            filename += f"_{v}-{getattr(config, k)}"
    filename += ".pkl"
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

        (all_obss, all_actions, _, _, _, all_dones) = dataset[:]

        self.rng_key = jax.random.PRNGKey(config.seed)
        self.selection_metric = config.selection_metric
        self.obss = all_obss.numpy()
        self.actions = all_actions.numpy()
        self.dones = all_dones.numpy()

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
            / f"{config.exp_name}_{config.model_ckpt}"
            / config.ckpt_dir
            / f"epoch_{config.ckpt_step}.pkl"
        )
        self.model_ckpt_path = model_ckpt_path

        obs_dim = self.obss.shape[-1]
        action_dim = self.actions.shape[-1]

        if self.selection_metric == "q_variance":
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_q(config, obs_dim, action_dim, None)
        elif (
            self.selection_metric == "policy_variance"
            or self.selection_metric == "influence_function"
        ):
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_bc(config, obs_dim, action_dim, None)

        logging.info(f"model ckpt path: {model_ckpt_path}")
        print(f"model ckpt path: {model_ckpt_path}")

        start_indices = np.where(self.dones)[0]
        start_indices += 1
        self.start_indices = np.insert(start_indices, 0, 0)

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
            logging.info(f"computing policy variance")
            policy_rng_keys = jax.random.split(
                self.rng_key, self.config.num_policies + 1
            )
            self.rng_key = policy_rng_keys[0]
            action_preds = jax.vmap(
                lambda param, rng_key: self.ts.apply_fn(param, rng_key, self.obss)
            )(self.ts.params, policy_rng_keys[1:])

            # compute variance between ensemble
            # average over ensemble and sum over action dim
            metric = jnp.var(action_preds, axis=0).sum(axis=-1)

        elif self.selection_metric == "influence_function":
            # this depends on what we want to compute influence of point wrt to

            # let's just use the first policy in ensemble
            params = jax.tree_map(lambda x: x[0], self.ts.params)
            self.ts = self.ts.replace(params=params)
            policy_rng, self.rng_key = jax.random.split(self.rng_key)
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
        logging.info(f"metric max: {np.max(metric)}, min: {np.min(metric)}")
        return metric

    def visualize_selected_states(
        self, selected_indices, selected_state_traj_indx, env=None
    ):
        num_states = len(selected_indices)
        num_rows_per_fig = self.config.max_states_visualize
        selected_states = self.obss[selected_indices]

        # chunk the selected states
        selected_states_chunk = np.array_split(
            selected_states, num_states // (num_rows_per_fig * 2)
        )
        selected_state_traj_indx_chunk = np.array_split(
            selected_state_traj_indx, num_states // (num_rows_per_fig * 2)
        )

        for chunk_indx, chunk in tqdm.tqdm(enumerate(selected_states_chunk)):
            imgs = []
            for state in chunk:
                obs, _ = env.reset_to_state(state)
                img = env.render()
                imgs.append(img)

            # visualized selected states to augment
            imgs = np.array(imgs)

            # import ipdb

            # ipdb.set_trace()

            # img, trajectory, metric, density if available
            n_plots_p_img = self.config.top_k * 2 + 1
            if self.config.reweight_with_density:
                n_plots_p_img += 1

            fig, axs = plt.subplots(
                num_rows_per_fig,
                n_plots_p_img,
                figsize=(4 * n_plots_p_img, 3 * num_rows_per_fig),
            )
            axs_flat = axs.flatten()

            min_x, max_x = np.min(self.obss[:, 0]), np.max(self.obss[:, 0])
            min_y, max_y = np.min(self.obss[:, 1]), np.max(self.obss[:, 1])

            # import ipdb

            # ipdb.set_trace()
            count = num_rows_per_fig * self.config.top_k

            for indx, row in enumerate(range(num_rows_per_fig)):
                # show the top k states
                for i in range(self.config.top_k):
                    ax = axs_flat[row * n_plots_p_img + i]
                    ax.axis("off")
                    ax.imshow(imgs[indx * self.config.top_k + i])

                ax1 = axs_flat[row * n_plots_p_img + self.config.top_k]

                # plot the trajectory and use metric as color
                traj_indx = selected_state_traj_indx_chunk[chunk_indx][
                    indx * self.config.top_k
                ]
                start = self.start_indices[traj_indx]
                end = self.start_indices[traj_indx + 1]
                traj = self.obss[start:end]

                metric_ = self.metric[start:end]
                reweighted_metric_ = np.array(self.reweighted_metric[start:end])

                ax1.set_xlim(min_x, max_x)
                ax1.set_ylim(min_y, max_y)

                # plot trajectory
                ax1.scatter(
                    traj[:, 0],
                    traj[:, 1],
                    c=np.array(metric_),
                    alpha=0.5,
                    cmap="viridis",
                    s=100,
                )

                # plot selected states
                for i in range(self.config.top_k):
                    ax1.scatter(
                        chunk[indx * self.config.top_k + i, 0],
                        chunk[indx * self.config.top_k + i, 1],
                        c="r",
                        s=150,
                    )

                # plot goal
                ax1.scatter(traj[0, 4], traj[0, 5], marker="x", c="g", s=150)

                for i in range(self.config.top_k):
                    ax2 = axs_flat[row * n_plots_p_img + self.config.top_k + 1 + i]
                    # plot the metric and reweighted metric
                    if self.config.reweight_with_density:
                        ax2.plot(metric_, label="metric", color="b")
                        ax2.plot(
                            reweighted_metric_, label="reweighted metric", color="r"
                        )

                        # get top i-th reweighted metric
                        top_i = np.argpartition(reweighted_metric_, -(i + 1))[
                            -(i + 1) :
                        ][0]

                        # import ipdb

                        # ipdb.set_trace()
                        ax2.axvline(
                            x=top_i,
                            color="r",
                            linestyle="--",
                            linewidth=3,
                        )
                    else:
                        ax2.plot(metric_, label="metric")

                    # ax2.legend()
                    top_i = np.argpartition(metric_, -(i + 1))[-(i + 1) :][0]
                    ax2.axvline(
                        x=top_i,
                        color="b",
                        linestyle="--",
                        linewidth=3,
                    )

                if self.config.reweight_with_density:
                    ax3 = axs_flat[row * n_plots_p_img + self.config.top_k * 2 + 1]
                    # plot density metric
                    densities = self.density_estimates[start:end]
                    densities = postprocess_density(densities, self.config.lamb)
                    ax3.plot(densities, label="density", color="g")

            col_names = (
                ["Image"] * self.config.top_k
                + ["Trajectory"]
                + ["Metric"] * self.config.top_k
                + ["Density"]
            )
            for ax, col in zip(axs[0], col_names):
                ax.set_title(col)

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # write to wandb_run
            if self.wandb_run:
                # save as image instead of plotly interactive figure
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                self.wandb_run.log({f"selected_states": wandb.Image(Image.open(buf))})

            # save figure
            plt.savefig(f"selected_states_{chunk_indx}.png")

    def select_states_per_traj(self):
        """
        Handles computation of the metric and reranking states based on metric.
        """
        # iterate over each trajectory and reweigh the metric
        selected_indices_traj = []
        selected_indices_global = []
        selected_state_traj_indx = []

        # pick top k states per trajectory
        for traj_indx in tqdm.tqdm(range(len(self.start_indices) - 1)):
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

        for indx in selected_indices_global:
            for traj_indx in range(len(self.start_indices) - 1):
                start = self.start_indices[traj_indx]
                end = self.start_indices[traj_indx + 1]
                if start <= indx < end:
                    selected_state_traj_indx.append(traj_indx)
                    selected_indices_traj.append(indx - start)
                    break

        return selected_indices_traj, selected_indices_global, selected_state_traj_indx


def run_augmentation(config, dataset, infos, wandb_run=None):
    config = ConfigDict(config)

    # update the model ckpt based on config
    config.model_ckpt = f"nt-{config.num_trajs}_s-{config.seed}"

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

    # if config.visualize:
    #     selector.visualize_selected_states(
    #         selected_indices_global, selected_state_traj_indx, env
    #     )

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
    selected_infos = [infos[ind] for ind in selected_indices_global]

    # import ipdb

    # ipdb.set_trace()

    # save images and generate a video of the augmentation dataset
    imgs = []

    # total number of new transitions = (# augmentations per state) x (# num total trajs) x (# expert steps)
    for indx, state in tqdm.tqdm(enumerate(selected_states)):

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
    augment_data_file = create_aug_filename(config)
    save_file = Path(config.data_dir) / "augment_datasets" / augment_data_file
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
            Path(config.data_dir)
            / "augment_datasets"
            / augment_data_file.replace(".pkl", ".mp4")
        )
        imageio.mimsave(
            video_file,
            imgs,
            fps=10,
        )

    env.close()


# run with ray tune
param_space = {
    # "seed": tune.grid_search([0, 1, 2]),
    # "num_trajs": tune.grid_search([5, 10, 25, 50, 75, 100, 125, 150, 175, 200]),
    "num_trajs": tune.grid_search([5, 10]),
}


def main(_):
    config = _CONFIG.value

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # STEP 1: Load base dataset and pass it into each run
    logging.info("Loading dataset")
    if config.env == "MAZE" or config.env == "MW":
        dataset, train_loader, test_loader, obs_dim, action_dim, infos = (
            load_pkl_dataset(
                config.data_dir,
                config.data_file,
                batch_size=1,
                num_trajs=10000,  # use all
                train_perc=1.0,
                env=config.env,
            )
        )
        obs_dim = obs_dim
        action_dim = action_dim
        logging.info(f"obs_dim: {obs_dim} action_dim: {action_dim}")
    else:
        raise NotImplementedError

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
            tune.with_parameters(
                run_augmentation_fn, dataset=dataset, infos=infos, wandb_run=wandb_run
            ),
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
        run_augmentation(config, dataset, infos)


if __name__ == "__main__":
    app.run(main)
