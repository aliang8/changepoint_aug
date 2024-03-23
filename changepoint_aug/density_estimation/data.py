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

    # split into train and test
    train_size = int(train_perc * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # convert for using with jax
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = NumpyLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    print("number of train batches: ", len(train_dataloader))
    print("number of test batches: ", len(test_dataloader))

    return (
        dataset,
        train_dataloader,
        test_dataloader,
        2,
        1,
    )


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


def load_pkl_dataset(
    data_dir: str,
    data_file: str,
    num_trajs: int,
    batch_size: int,
    train_perc: float = 1.0,
    env: str = "MAZE",
    augmentation_data_files: List[str] = [],
):
    data_file = Path(data_dir) / data_file
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # data is a list of tuples
    # use done to determine when it ends
    rollouts = []
    rollout = []
    for t, transition in enumerate(data):
        obs_t, obs_tp1, action, rew, terminated, truncated, info = transition
        if env == "MW":
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
    for data_file in augmentation_data_files:
        logging.info(f"loading augmentation file: {data_file}")
        data_file = Path(data_dir) / data_file
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        metadata = data["metadata"]
        data = data["rollouts"]

        # split data into list
        rollout = [data[i : i + 10] for i in range(0, len(data), 10)]
        logging.info(f"num augmentation rollouts: {len(rollout)}")
        augmentation_rollouts.extend(rollout)

    obs_data = []
    obs_tp1_data = []
    action_data = []
    action_tp1_data = []
    rew_data = []
    done_data = []

    # select subset of random trajectories
    traj_indices = np.random.choice(
        len(rollouts), min(len(rollouts), num_trajs), replace=False
    )
    rollouts = [rollouts[i] for i in traj_indices.tolist()]
    logging.info(f"number of selected rollouts base: {len(rollouts)}")

    # add augmentation data
    rollouts.extend(augmentation_rollouts)

    for rollout in rollouts:
        obs = [step[0] for step in rollout]
        if env == "MAZE":
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

    # convert to torch tensors
    obs_data = torch.from_numpy(np.array(obs_data)).squeeze()
    obs_tp1_data = torch.from_numpy(np.array(obs_tp1_data)).squeeze()
    action_data = torch.from_numpy(np.array(action_data))
    action_tp1_data = torch.from_numpy(np.array(action_tp1_data))
    rew_data = torch.from_numpy(np.array(rew_data))
    done_data = torch.from_numpy(np.array(done_data))

    # tensor_dict = {
    #     "obs_data": obs_data,
    #     "obs_tp1_data": obs_tp1_data,
    #     "action_data": action_data,
    #     "action_tp1_data": action_tp1_data,
    #     "rew_data": rew_data,
    #     "done_data": done_data,
    # }

    logging.info(
        f"obs_data shape: {obs_data.shape} action_data shape: {action_data.shape}"
    )
    logging.info(
        f"min obs data: {obs_data.min(axis=0)}, max obs data: {obs_data.max(axis=0)}"
    )
    logging.info(
        f"min action data: {action_data.min(axis=0)}, max action data: {action_data.max(axis=0)}"
    )

    # data = TensorDict(tensor_dict, batch_size=len(obs_data))

    # create dataloader
    dataset = torch.utils.data.TensorDataset(
        obs_data, action_data, obs_tp1_data, action_tp1_data, rew_data, done_data
    )

    # split into train and test
    train_size = int(train_perc * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # convert for using with jax
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = NumpyLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    batch = next(iter(train_dataloader))
    logging.info(batch[0][0])

    # import ipdb

    # ipdb.set_trace()

    logging.info(f"number of train batches: {len(train_dataloader)}")
    logging.info(f"number of test batches: {len(test_dataloader)}")

    return (
        dataset,
        train_dataloader,
        test_dataloader,
        obs_data.shape[-1],
        action_data.shape[-1],
    )
