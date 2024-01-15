import pathlib
import time
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils import data
import tqdm
import click
from pathlib import *
from influence_functions import (
    BaseObjective,
    CGInfluenceModule,
    AutogradInfluenceModule,
)

# load dataset
from imitation.data import serialize
import os
from imitation.algorithms import base as algo_base
from imitation.algorithms.bc import reconstruct_policy
from bc import CustomPolicy


@click.command()
@click.option("--env_name", default="metaworld-assembly-v2")
@click.option("--root_dir", default="/scr/aliang80/changepoint_aug")
@click.option("--dataset_file", default="datasets/expert_dataset/assembly-v2_50")
@click.option("--num_demos", default="25")
@click.option("--n_eval_episodes", default=25)
@click.option("--num_videos_save", default=3)
@click.option("--num_bc_epochs", default=100)
@click.option("--seed", default=0)
def main(
    env_name: str,
    num_demos: int,
    root_dir: str,
    dataset_file: str,
    n_eval_episodes: int,
    num_videos_save: int,
    num_bc_epochs: int = 100,
    seed: int = 0,
):
    # load dataset
    print("load dataset from ", dataset_file)
    dataset_file = os.path.join(root_dir, dataset_file)
    expert_trajectories = serialize.load(dataset_file)
    print("number of expert trajectories: ", len(expert_trajectories))

    # compute the total number of transitions in dataset
    dl = algo_base.make_data_loader(expert_trajectories, batch_size=1)
    total_transitions = len(dl.dataset)
    print("total number of transitions: ", total_transitions)

    # sample a subset of the dataset for holdout test
    num_test = int(0.2 * total_transitions)

    ckpt_path = (
        Path(root_dir)
        / "bc_policies"
        / env_name
        / f"demos_{num_demos}"
        / f"e_{num_bc_epochs}"
        / f"s_{seed}.zip"
    )

    # load trained policy
    print("load policy from ", ckpt_path)
    trained_policy = reconstruct_policy(policy_path=ckpt_path, device="cuda")

    ent_weight = 1e-3
    l2_weight = 0

    # ===========
    # Initialize influence module using custom objective
    # ===========
    class BCObjective(BaseObjective):
        def train_outputs(self, model, batch):
            tensor_obs = batch["obs"]
            acts = batch["acts"]
            (_, log_prob, entropy) = model.evaluate_actions(
                tensor_obs,  # type: ignore[arg-type]
                acts,
            )
            return log_prob, entropy

        def train_loss_on_outputs(self, outputs, batch):
            log_prob, entropy = outputs
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            entropy = entropy.mean() if entropy is not None else None
            ent_loss = ent_weight * (entropy if entropy is not None else torch.zeros(1))
            neglogp = -log_prob
            return neglogp

        def train_regularization(self, params):
            l2_norms = [torch.sum(torch.square(w)) for w in params]
            l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
            # sum of list defaults to float(0) if len == 0.
            assert isinstance(l2_norm, torch.Tensor)
            return l2_weight * l2_norm

        def test_loss(self, model, params, batch):
            tensor_obs = batch["obs"]
            acts = batch["acts"]
            (_, log_prob, entropy) = model.evaluate_actions(
                tensor_obs,  # type: ignore[arg-type]
                acts,
            )
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            entropy = entropy.mean() if entropy is not None else None

            l2_norms = [torch.sum(torch.square(w)) for w in params]
            l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
            # sum of list defaults to float(0) if len == 0.
            assert isinstance(l2_norm, torch.Tensor)

            ent_loss = ent_weight * (entropy if entropy is not None else torch.zeros(1))
            neglogp = -log_prob
            l2_loss = l2_weight * l2_norm
            loss = neglogp + ent_loss + l2_loss
            return loss

    dl = algo_base.make_data_loader(expert_trajectories, batch_size=1)
    data = next(iter(dl))
    print(data.keys())
    # print(dl.dataset)
    # print(dl.dataset[0])

    # only compute influence wrt to the final action layer of the model
    class CustomAutogradInfluenceModule(AutogradInfluenceModule):
        # only with respect to the last layer of the model
        def _model_params(self, with_names=True):
            assert not self.is_model_functional
            return tuple(
                (name, p) if with_names else p
                for name, p in self.model.named_parameters()
                if p.requires_grad and "action_net" in name
            )

    # the inverse hess should be computed during init
    if os.path.exists("inv_hess.npy"):
        print("loading inverse hessian from file")
        inv_hess = np.load("inv_hess.npy")
    else:
        inv_hess = module.inverse_hess

        # save this too
        np.save("inv_hess.npy", inv_hess.cpu().numpy())

    module = CustomAutogradInfluenceModule(
        model=trained_policy,
        objective=BCObjective(),
        train_loader=algo_base.make_data_loader(expert_trajectories, batch_size=256),
        test_loader=algo_base.make_data_loader(expert_trajectories, batch_size=256),
        device="cuda",
        damp=0.001,
        inv_hess=inv_hess,
        # atol=1e-8,
        # maxiter=1000,
    )

    stest = None
    # then compute influence
    all_train_idxs = np.arange(total_transitions).tolist()

    # let's try averaging over 10 random test sets
    all_influences = []
    for _ in range(5):
        test_idxs = np.random.choice(all_train_idxs, size=num_test, replace=False)
        # print(test_idxs)
        subset_all_train_idxs = np.arange(200).tolist()
        influences = module.influences(
            train_idxs=subset_all_train_idxs, test_idxs=test_idxs, stest=stest
        )
        all_influences.append(influences)

    all_influences = torch.cat(all_influences).mean(dim=0)
    print(torch.argsort(influences))


if __name__ == "__main__":
    main()
