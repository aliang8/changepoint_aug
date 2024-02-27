import os
from functools import partial
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ActionModel(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=None):
        super(ActionModel, self).__init__()
        self.units = neurons if neurons is not None else [300]

        layer_list = [
            nn.Linear(input_dim, self.units[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(len(self.units) - 1):
            layer_list.append(nn.Linear(self.units[i], self.units[i + 1]))
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Linear(self.units[-1], output_dim))

        self.main = nn.Sequential(*layer_list)
        self.max_loss, self.min_loss = 0, 0

    def forward(self, input):
        output = self.main(input)
        return output

    def load_margin(self, min_loss, max_loss):
        self.min_loss = min_loss
        self.max_loss = max_loss


class ActionDataset(Dataset):
    def __init__(self, actions, prev_actions, curr_actions):
        self.actions = actions
        self.prev_actions, self.curr_actions = prev_actions, curr_actions
        self.stack_size = self.prev_actions + self.curr_actions
        self.all_actions = []
        # self.all_steers, self.all_throttles, self.all_brakes = [], [], []

        # img_path_list, measurements = np.load(path)
        # for index in range(len(img_path_list)):
        self.indices = []
        for traj_indx, traj in enumerate(self.actions):
            for timestep in range(len(traj)):
                start_index = timestep - prev_actions
                end_index = timestep + (curr_actions - 1)
                if start_index >= 0:
                    print(start_index, end_index)

                    if end_index < len(traj):
                        action_list = []
                        for idx in range(start_index, end_index + 1):
                            action_list.append(traj[idx])
                            # action_list.append(measurements[idx]["steer"])
                            # action_list.append(measurements[idx]["throttle"])
                            # action_list.append(measurements[idx]["brake"])
                        # print(len(action_list))
                        self.all_actions.append(np.array(action_list))
                        self.indices.append([traj_indx, timestep])
                    # self.all_steers.append(measurements[index]["steer"])
                    # self.all_throttles.append(measurements[index]["throttle"])
                    # self.all_brakes.append(measurements[index]["brake"])

        # self.mean = np.array(
        #     [
        #         np.mean(self.all_steers),
        #         np.mean(self.all_throttles),
        #         np.mean(self.all_brakes),
        #     ]
        # )
        # self.std = np.array(
        #     [
        #         np.std(self.all_steers),
        #         np.std(self.all_throttles),
        #         np.std(self.all_brakes),
        #     ]
        # )
        # self.stacked_mean = np.tile(self.mean, self.stack_size)
        # self.stacked_std = np.tile(self.std, self.stack_size)

        for index in range(len(self.all_actions)):
            stacked_actions = self.all_actions[index]
            self.all_actions[index] = torch.FloatTensor(stacked_actions)
            self.indices[index] = torch.LongTensor(self.indices[index])
        # print(self.all_actions[0].shape)

    def __getitem__(self, idx):
        return (
            self.all_actions[idx][: -1 * self.curr_actions].flatten(),
            self.all_actions[idx][-1 * self.curr_actions :].flatten(),
            self.indices[idx],
        )

    def __len__(self):
        return len(self.all_actions)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std


def weighted_loss(output, target, weight=None):
    loss = (output - target).pow(2)
    if weight is not None:
        loss = loss * weight
        return loss.sum(dim=-1)
    else:
        return loss.mean(dim=-1)


def train(model, optimizer, train_loader, loss_func, epoch):
    model.train()
    optimizer.zero_grad()
    accumulate_loss = []
    for batch_idx, (batch_x, batch_y, _) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        # print(batch_x.shape, batch_y.shape)
        model.zero_grad()
        output = model(batch_x)
        # print(output.shape)
        loss = loss_func(output, batch_y).mean()
        # print(loss)
        accumulate_loss.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # logger.add_scalar("train_loss", np.mean(accumulate_loss), epoch)
    print("iter: {} train loss: {}".format(epoch, np.mean(accumulate_loss)))


def adjustlr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_rate


def train_ape_model(
    actions, prev_actions, curr_actions, epoch_num, model_layer_neurons, if_save
):
    identifier = str(int(datetime.utcnow().timestamp())) + "".join(
        [str(np.random.randint(10)) for _ in range(8)]
    )

    action_dim = actions[0][0].shape[-1]
    print("action dim: ", action_dim)
    dataset = ActionDataset(
        actions=actions,
        prev_actions=prev_actions,
        curr_actions=curr_actions,
    )
    x, y, _ = dataset.__getitem__(0)
    print(x.shape, y.shape)
    # dataset = ActionDataset(data_path, prev_actions, curr_actions)
    # mean, std = dataset.get_mean(), dataset.get_std()

    prev_seed = torch.get_rng_state()
    torch.manual_seed(0)
    trainset, testset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )
    torch.set_rng_state(prev_seed)

    trainloader = DataLoader(dataset=trainset, batch_size=512, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=512, shuffle=False)

    model = ActionModel(
        input_dim=prev_actions * action_dim,
        output_dim=curr_actions * action_dim,
        neurons=model_layer_neurons,
    )
    model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_func = partial(
        weighted_loss,
        weight=torch.Tensor([0.25, 0.25, 0.25, 0.25] * curr_actions).cuda(),
    )

    layer_num_str = [str(n) for n in model_layer_neurons]
    result_dir = "/scr/aliang80/changepoint_aug/results/action_correlation/prev{}-curr{}-layer{}-{}".format(
        prev_actions, curr_actions, "-".join(layer_num_str), identifier
    )
    save_dir = "/scr/aliang80/changepoint_aug/results/action_correlation/checkpoints/prev{}-curr{}-layer{}.pkl".format(
        prev_actions, curr_actions, "-".join(layer_num_str)
    )

    # log_dir = os.path.join(result_dir, "run")
    # os.makedirs(log_dir)

    # writer = SummaryWriter(log_dir)

    min_loss, max_loss = 0, 0
    for epoch in range(1, epoch_num + 1):
        train(model, optimizer, trainloader, loss_func, epoch)
        # min_loss, max_loss = test(model, testloader, loss_func, epoch, writer)

        if epoch % (epoch_num // 3) == 0:
            adjustlr(optimizer, 0.1)

    if if_save:
        # os.makedirs("results/action_correlation", exist_ok=True)
        # os.makedirs(results_dir, exist_ok=True)
        print(os.path.dirname(save_dir))
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        checkpoint = {
            "state_dict": model.state_dict(),
            "min_loss": min_loss,
            "max_loss": max_loss,
            # "mean": mean,
            # "std": std,
        }
        torch.save(checkpoint, save_dir)

    return model
