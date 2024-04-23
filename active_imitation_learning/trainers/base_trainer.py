"""
Base Trainer

Usage:
"""

from absl import app, logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import flax
import torch
import wandb
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from pathlib import Path
from tensorflow_probability.substrates import jax as tfp

from active_imitation_learning.utils.data import (
    load_maze_dataset,
    make_blob_dataset,
    NumpyLoader,
    parition_batch_train_test,
    split_data_by_traj,
    concatenate_batches,
)

dist = tfp.distributions


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        # set torch seed to maintain reproducibility
        logging.info(f"setting seed: {config.seed}")
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="active_imitation_learning",
                name=config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                group=config.group_name if config.group_name else None,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # load dataset
        logging.info("loading dataset")
        self.dataset, self.train_loader, self.test_loader = self.load_data()
        logging.info(f"done loading dataset")

        # store initial conditions for evaluation
        if config.run_ic_evals:
            logging.info("keeping track of the ICs for evaluation")
            start_indices = np.where(self.dataset["dones"])[0]
            start_indices += 1
            start_indices = np.insert(start_indices, 0, 0)
            start_indices = start_indices[:-1]
            self.initial_conditions = np.asarray(
                self.dataset["observations"][start_indices]
            )
            self.initial_conditions = self.initial_conditions[
                : self.config.base_num_trajs
            ]

            # add perturbations to the initial conditions
            all_ics = []
            if config.perturbations_per_trajs > 0:
                for indx in range(len(self.initial_conditions)):
                    ic = self.initial_conditions[indx]
                    all_ics.append(ic)
                    for _ in range(config.perturbations_per_trajs):
                        new_ic = ic.copy()
                        # perturb the x,y location
                        noise = np.random.normal(0, 0.1, size=(4,))
                        new_ic[:2] += noise[:2]
                        # we can also perturb the goal location at 4, 5
                        new_ic[4:6] += noise[2:4]
                        all_ics.append(new_ic)
            self.initial_conditions = all_ics
            logging.info(
                f"number of initial conditions: {len(self.initial_conditions)}"
            )

        self.obs_shape = self.dataset["observations"].shape[1:]
        self.action_dim = self.dataset["actions"].shape[-1]

        self.root_dir = Path(self.config.root_dir)

        # setup log dirs
        self.ckpt_dir = self.root_dir / self.config.ckpt_dir
        logging.info(f"ckpt_dir: {self.ckpt_dir}")

        # make it
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.root_dir / self.config.video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if config.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def load_data(self):
        if self.config.env == "MAZE" or self.config.env == "MW":
            dataset = load_maze_dataset(self.config)
        elif self.config.env == "TOY":
            dataset, train_loader, test_loader, obs_dim, action_dim = make_blob_dataset(
                10000,
                # centers=3,
                n_features=2,
                random_state=0,
                centers=[[0, 2], [-1, -1], [1, -1]],
            )
        elif self.config.env == "D4RL":
            import d4rl
            import gym

            env = gym.make(self.config.env_id).unwrapped
            dataset = d4rl.qlearning_dataset(env)

            # import ipdb

            # ipdb.set_trace()

            dataset = dict(
                observations=dataset["observations"],
                actions=dataset["actions"],
                next_observations=dataset["next_observations"],
                rewards=dataset["rewards"],
                dones=dataset["terminals"].astype(np.float32),
            )
            dataset["rewards"] = (
                dataset["rewards"] * self.config.reward_scale + self.config.reward_bias
            )
            dataset["actions"] = np.clip(
                dataset["actions"], -self.config.clip_action, self.config.clip_action
            )
            traj_dataset = split_data_by_traj(
                dataset, max_traj_length=self.config.max_episode_steps
            )
            logging.info(f"number of trajectories: {len(traj_dataset)}")

            # random select a subset of trajectories
            indices = np.arange(len(traj_dataset))
            selected_indices = np.random.choice(
                indices, self.config.num_trajs, replace=False
            )
            logging.info(f"selected indices: {selected_indices}")
            traj_dataset = [traj_dataset[i] for i in selected_indices]
            logging.info(
                f"avg returns: {np.mean([np.sum(traj['rewards']) for traj in traj_dataset])}"
            )
            logging.info(
                f"avg normalized returns: {np.mean([env.get_normalized_score(np.sum(traj['rewards'])) for traj in traj_dataset])}"
            )
            # import ipdb

            # ipdb.set_trace()
            dataset = concatenate_batches(traj_dataset)

        dataset = {k: torch.from_numpy(v) for k, v in dataset.items()}

        train_dataset, test_dataset = parition_batch_train_test(
            dataset, train_ratio=self.config.train_perc
        )

        logging.info(f"dataset keys: {train_dataset.keys()}")
        logging.info(f"train dataset obs shape: {train_dataset['observations'].shape}")
        logging.info(f"train dataset action shape: {train_dataset['actions'].shape}")
        logging.info(f"test dataset obs shape: {test_dataset['observations'].shape}")

        # import ipdb

        # ipdb.set_trace()
        keys = [
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "dones",
        ]
        train_dataset = torch.utils.data.TensorDataset(
            *[train_dataset[k] for k in keys]
        )
        test_dataset = torch.utils.data.TensorDataset(*[test_dataset[k] for k in keys])

        # convert for using with jax
        train_dataloader = NumpyLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_dataloader = NumpyLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
        )

        logging.info(
            f"num train batches: {len(train_dataloader)}, num test batches: {len(test_dataloader)}"
        )

        return dataset, train_dataloader, test_dataloader

    def train(self):
        # run one eval before training
        if not self.config.skip_first_eval:
            test_metrics = self.test(0)

        for epoch in tqdm.tqdm(
            range(1, self.config.num_epochs + 1),
            disable=self.config.disable_tqdm,
            desc="epochs",
        ):
            # run a test first
            if epoch % self.config.test_interval == 0:
                test_time = time.time()
                test_metrics = self.test(epoch)
                test_total_time = time.time() - test_time

                # save based on key
                # import ipdb

                # ipdb.set_trace()
                if self.config.save_key in test_metrics:
                    key = self.config.save_key
                    if (
                        self.config.best_metric == "max"
                        and test_metrics[key] > self.best_metric
                    ) or (
                        self.config.best_metric == "min"
                        and test_metrics[key] < self.best_metric
                    ):
                        self.best_metric = test_metrics[key]
                        ckpt_file = self.ckpt_dir / f"best.pkl"
                        logging.info(
                            f"new best value: {test_metrics[key]}, saving best model at epoch {epoch} to {ckpt_file}"
                        )
                        with open(ckpt_file, "wb") as f:
                            pickle.dump(self.ts.params, f)

                        # create a file with the best metric in the name, use a placeholder
                        best_ckpt_file = self.ckpt_dir / "best.txt"
                        with open(best_ckpt_file, "w") as f:
                            f.write(f"{epoch}, {test_metrics[key]}")

                if self.wandb_run is not None:
                    self.wandb_run.log(test_metrics)
                    self.wandb_run.log({"time/test_time": test_total_time})

            epoch_time = time.time()
            # for batch in tqdm.tqdm(
            #     self.train_loader,
            #     disable=self.config.disable_tqdm,
            #     desc="iterating over batches",
            # ):
            for batch in self.train_loader:
                batch_time = time.time()
                train_metrics = self.train_step(batch)
                batch_total_time = time.time() - batch_time

                self.global_step += 1

                if self.wandb_run is not None:
                    # add prefix to metrics
                    train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                    self.wandb_run.log(train_metrics)
                    self.wandb_run.log({"time/batch_time": batch_total_time})
                    print(train_metrics)

            epoch_total_time = time.time() - epoch_time

            # log time information
            if self.wandb_run is not None:
                self.wandb_run.log({"time/epoch_time": epoch_total_time})

            # save model at set intervals
            if epoch % self.config.save_interval == 0:
                ckpt_file = self.ckpt_dir / f"epoch_{epoch}.pkl"
                logging.info(f"saving model at epoch {epoch} to {ckpt_file}")
                with open(ckpt_file, "wb") as f:
                    pickle.dump(self.ts.params, f)
