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

sys.path.append("/scr/aliang80/changepoint_aug/changepoint_aug/old")
from sac_torch import Actor
from sac_eval import make_env as make_env_sac

from changepoint_aug.density_estimation.data import load_pkl_dataset, make_blob_dataset
from changepoint_aug.density_estimation.trainers.q_sarsa import create_ts as create_ts_q
from changepoint_aug.density_estimation.trainers.cvae import create_ts as create_ts_cvae
from changepoint_aug.density_estimation.trainers.bc import create_ts as create_ts_bc
from changepoint_aug.density_estimation.visualize import estimate_density
from changepoint_aug.density_estimation.utils import make_env

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MUJOCO_EGL"] = "egl"


_CONFIG = config_flags.DEFINE_config_file("config")


def postprocess_density(densities):
    """
    Postprocess density estimates to be between 0 and 1
    """
    densities = densities - np.min(densities)
    densities = densities / np.max(densities)
    densities *= 2

    # should i also apply a smoothing kernel?
    densities = np.convolve(densities, np.ones(5) / 5, mode="same")
    return densities


class QueryStateSelector:
    def __init__(self, config: FrozenConfigDict, dataset, wandb_run=None):
        # print current directory
        print("current dir: ", os.getcwd())

        self.config = config
        self.wandb_run = wandb_run

        (all_obss, all_actions, _, _, _, all_dones) = dataset[:]

        self.rng_key = jax.random.PRNGKey(config.seed)
        self.metric = config.selection_metric
        self.obss = all_obss.numpy()
        self.actions = all_actions.numpy()
        self.dones = all_dones.numpy()

        assert self.metric in ["q_variance", "policy_variance"]

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
            self.density_ts = create_ts_cvae(config, 2, None)
        else:
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

        obs_dim = self.obss.shape[-1]
        action_dim = self.actions.shape[-1]

        if self.metric == "q_variance":
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_q(config, obs_dim, action_dim, None)
        elif self.metric == "policy_variance":
            config.load_from_ckpt = str(model_ckpt_path)
            self.ts = create_ts_bc(config, obs_dim, action_dim, None)

        logging.info(f"model ckpt path: {model_ckpt_path}")
        print(f"model ckpt path: {model_ckpt_path}")

    def select_states(self, env=None):
        """
        Handles computation of the metric and reranking states based on metric.
        """

        # STEP 1: compute density for every obss and corresponding goal
        # STEP 2: compute metric for every obss
        # STEP 3: reweigh metric by density (optional)
        # STEP 4: select top k states

        start_indices = np.where(self.dones)[0]
        start_indices += 1
        start_indices = np.insert(start_indices, 0, 0)
        # start_indices = start_indices[:-1]

        if self.config.reweight_with_density:
            logging.info(f"computing density estimates")
            density_estimates = []

            chunk_size = 1000
            rng_keys = jax.random.split(self.rng_key, self.obss.shape[0] + 1)
            self.rng_key = rng_keys[0]
            obss_chunked = np.array_split(self.obss, self.obss.shape[0] // chunk_size)
            rng_keys_chunked = np.array_split(
                rng_keys[1:], self.obss.shape[0] // chunk_size
            )

            # STEP 1
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

        # STEP 2
        if self.metric == "q_variance":
            pass
        elif self.metric == "policy_variance":
            logging.info(f"computing policy variance")
            policy_rng_keys = jax.random.split(
                self.rng_key, self.config.num_policies + 1
            )
            self.rng_key = policy_rng_keys[0]
            action_preds = jax.vmap(
                lambda param, rng_key: self.ts.apply_fn(param, self.rng_key, self.obss)
            )(self.ts.params, policy_rng_keys[1:])

            # compute variance between ensemble
            # average over ensemble and sum over action dim
            metric = jnp.var(action_preds, axis=0).sum(axis=-1)
            logging.info(f"done computing policy variance, {metric.shape}")

        # STEP 3

        # iterate over each trajectory and reweigh the metric
        reweighted_metrics = []

        selected_states = []
        selected_indices = []
        selected_state_traj_indx = []

        for traj_indx in tqdm.tqdm(range(len(start_indices) - 1)):
            start = start_indices[traj_indx]
            end = start_indices[traj_indx + 1]
            metric_ = metric[start:end]

            # pick indx with max reweighted metric
            if self.config.reweight_with_density:
                densities = density_estimates[start:end]
                densities = postprocess_density(densities)
                reweighted_metric = metric_ * densities
                reweighted_metrics.extend(reweighted_metric)

                selected_indx = np.argpartition(reweighted_metric, -self.config.top_k)[
                    -self.config.top_k :
                ]
                # import ipdb

                # ipdb.set_trace()
            else:
                selected_indx = np.argpartition(metric_, -self.config.top_k)[
                    -self.config.top_k :
                ]

            selected_states.append(np.take(self.obss[start:end], selected_indx, axis=0))
            selected_indices.extend(selected_indx)
            selected_state_traj_indx.extend([traj_indx] * selected_indx.shape[0])

        selected_states = np.concatenate(selected_states)
        logging.info(f"number of selected states: {len(selected_indices)}")

        # STEP 4
        # TODO: maybe edit to select global top k states

        num_states = len(selected_indices)
        num_rows_per_fig = self.config.max_states_visualize

        # chunk the selected states
        selected_states_chunk = np.array_split(
            selected_states, num_states // num_rows_per_fig
        )
        selected_state_traj_indx_chunk = np.array_split(
            selected_state_traj_indx, num_states // num_rows_per_fig
        )

        for chunk_indx, chunk in tqdm.tqdm(enumerate(selected_states_chunk)):
            imgs = []
            for state in chunk:
                obs, _ = env.reset_to_state(state)
                img = env.render()
                imgs.append(img)

            # visualized selected states to augment
            imgs = np.array(imgs)

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
                start = start_indices[traj_indx]
                end = start_indices[traj_indx + 1]
                traj = self.obss[start:end]

                metric_ = metric[start:end]
                reweighted_metric_ = np.array(reweighted_metrics[start:end])

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
                    densities = density_estimates[start:end]
                    densities = postprocess_density(densities)
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

        return selected_states


def run_augmentation(config, dataset, wandb_run=None):
    config = ConfigDict(config)

    # update the model ckpt based on config
    config.model_ckpt = f"nt-{config.num_trajs}_s-{config.seed}"

    env = make_env(
        config.env,
        config.env_id,
        config.seed,
        max_episode_steps=config.max_episode_steps,
    )
    obs, _ = env.reset(seed=config.seed)
    # print(dataset[:][0][0])
    # obs, _ = env.reset_to_state(dataset[:][0][0].numpy())

    # STEP 2: Query states to augment
    # pick which states to query based on metric
    selector = QueryStateSelector(config, dataset, wandb_run=wandb_run)
    selected_states = selector.select_states(env)
    selected_states = selected_states[:100]

    # STEP 3: Augment selected states with a few steps of expert actions
    # Perturb states a little bit

    # load expert model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load(
    #     "/scr/aliang80/changepoint_aug/changepoint_aug/old/model_ckpts/sac_maze_rand_1000000.pt",
    #     map_location=device,
    # )
    # envs = gym.vector.SyncVectorEnv([make_env_sac(None, 0, 0, False, "")])
    # actor = Actor(envs).to(device)
    # actor.load_state_dict(state_dict["actor"])
    # actor.eval()

    # rollouts = []

    # for state in tqdm.tqdm(selected_states):
    #     obs, _ = env.reset_to_state(state)

    #     # take some perturbation steps
    #     for _ in range(config.num_perturb_steps):
    #         action = env.action_space.sample()
    #         next_obs, reward, done, truncated, info = env.step(action)
    #         obs = next_obs

    #     obs_input = np.concatenate(
    #         [obs["observation"], obs["desired_goal"], obs["meta"]],
    #     )
    #     obs_input = obs_input[None]

    #     for _ in range(config.num_expert_steps_aug):
    #         # don't use the sampled action
    #         action_sample, _, action_mean = actor.get_action(
    #             torch.Tensor(obs_input).to(device)
    #         )
    #         action = action_mean[0].detach().cpu().numpy()
    #         next_obs, reward, done, truncated, info = env.step(action)
    #         rollouts.append((obs, next_obs, action, reward, done, truncated, info))

    #         obs = next_obs

    # logging.info(f"number of new transitions: {len(rollouts)}")

    # # STEP 4: Save augmented dataset
    # save_file = Path(config.data_dir) / config.augment_data_file
    # logging.info(f"Saving augmented dataset to {save_file}")
    # with open(save_file, "wb") as f:
    #     pickle.dump(rollouts, f)

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
        dataset, train_loader, test_loader, obs_dim, action_dim = load_pkl_dataset(
            config.data_dir,
            config.data_file,
            batch_size=1,
            num_trajs=10000,  # use all
            train_perc=1.0,
            env=config.env,
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
                run_augmentation_fn, dataset=dataset, wandb_run=wandb_run
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
        run_augmentation(config, dataset)


if __name__ == "__main__":
    app.run(main)
