from absl import logging
import pickle
import numpy as np
import torch
import jax
from sklearn.datasets import make_blobs
import torch.utils.data
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple
from pathlib import Path


def split_data_by_traj(data, max_traj_length):
    dones = data["dones"].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate(
            [batch[key] for batch in batches], axis=0
        ).astype(np.float32)
    return concatenated


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch["observations"].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["observations"].shape[0], size=size)
    return index_batch(batch, indices)


def make_blob_dataset(
    n_samples, centers, n_features, random_state, train_perc: float, batch_size: int
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    return dataset


def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, torch.utils.data.default_collate(batch))


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def load_maze_dataset(config):
    data_file = Path(config.data_dir) / config.data_file
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # data is a list of tuples
    # use done to determine when it ends
    rollouts = []
    rollout = []
    for t, transition in enumerate(data):
        obs_t, obs_tp1, action, rew, terminated, truncated, info = transition
        if config.env == "MW":
            terminated = info["success"][0]
        rollout.append((obs_t, obs_tp1, action, rew, terminated, truncated, info))

        if terminated or truncated:
            rollouts.append(rollout)
            rollout = []

    logging.info(f"total transitions in base dataset: {len(data)}")

    num_rollouts = len(rollouts)
    logging.info(f"number of rollouts in base dataset: {num_rollouts}")
    avg_rollout_len = sum(len(r) for r in rollouts) / len(rollouts)
    logging.info(f"average length of base rollout: {avg_rollout_len}")

    augmentation_rollouts = []
    total_aug_transitions = 0
    for data_file in config.augmentation_data:
        data_file = (
            Path(config.data_dir) / "augment_datasets" / data_file / "dataset.pkl"
        )
        logging.info(f"loading augmentation file: {data_file}")

        with open(data_file, "rb") as f:
            data = pickle.load(f)

        metadata = data["metadata"]
        data = data["rollouts"]
        steps = metadata["config"]["num_expert_steps_aug"]
        num_transitions = metadata["num_transitions"]
        total_aug_transitions += num_transitions
        logging.info(
            f"loading augmentation dataset, steps: {steps}, num transitions: {num_transitions}"
        )

        # split data into list
        rollout = [data[i : i + steps] for i in range(0, len(data), steps)]
        logging.info(f"num augmentation rollouts: {len(rollout)}")
        augmentation_rollouts.extend(rollout)

    obs_data = []
    obs_tp1_data = []
    action_data = []
    action_tp1_data = []
    rew_data = []
    done_data = []
    info_data = []

    # select subset of random trajectories for base
    traj_indices = np.random.choice(
        len(rollouts), min(len(rollouts), config.base_num_trajs), replace=False
    )
    traj_indices = traj_indices.tolist()
    logging.info(f"old: {traj_indices}")

    # additional new trajs
    if config.num_additional_trajs > 0:
        for _ in range(config.num_shuffles):
            new_traj_indices = np.random.choice(
                len(rollouts), config.num_additional_trajs, replace=False
            )
        logging.info(f"new: {new_traj_indices}")
        traj_indices.extend(new_traj_indices.tolist())

    rollouts = [rollouts[i] for i in traj_indices]
    # import ipdb

    # ipdb.set_trace()
    num_base_transitions = 0
    for rollout in rollouts:
        num_base_transitions += len(rollout)

    logging.info(f"number of base trajectories: {len(rollouts)}")
    logging.info(f"number of base transitions: {num_base_transitions}")

    if len(augmentation_rollouts) > 0:
        # add augmentation data
        # subsample augmentation rollouts to be num_augmentation_steps
        num_transitions = (
            config.num_augmentation_steps
            if config.num_augmentation_steps > 0
            else total_aug_transitions
        )
        num_aug_trajs = int(num_transitions // len(augmentation_rollouts[0]))
        logging.info(f"num aug trajs available: {num_aug_trajs}")

        aug_traj_indices = np.random.choice(
            len(augmentation_rollouts),
            min(len(augmentation_rollouts), num_aug_trajs),
            replace=False,
        )
        augmentation_rollouts = [
            augmentation_rollouts[i] for i in aug_traj_indices.tolist()
        ]
        logging.info(f"num aug trajs used: {num_aug_trajs}")

        rollouts.extend(augmentation_rollouts)

    for rollout in rollouts:
        obs = [step[0] for step in rollout]
        if config.env == "MAZE":
            if isinstance(obs[0], np.ndarray):
                obs[0] = obs[0][0]

            if "meta" in obs[0]:
                obs = [
                    np.concatenate((o["observation"], o["desired_goal"], o["meta"]))
                    for o in obs
                ]
            else:
                obs = [
                    np.concatenate((o["observation"], o["desired_goal"])) for o in obs
                ]

        obs_data.extend(obs[:-1])
        obs_tp1_data.extend(obs[1:])
        action_data.extend([step[2] for step in rollout][:-1])
        action_tp1_data.extend([step[2] for step in rollout][1:])
        done = [step[4] for step in rollout][1:]
        done[-1] = True
        done_data.extend(done)
        rew_data.extend([step[3] for step in rollout][1:])
        info_data.extend([step[-1] for step in rollout])

    dataset = {
        "observations": np.array(obs_data),
        "actions": np.array(action_data),
        "next_observations": np.array(obs_tp1_data),
        "rewards": np.array(rew_data),
        "dones": np.array(done_data),
    }
    return dataset
