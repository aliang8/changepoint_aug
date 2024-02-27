from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

# import thrifty.algos.core as core
# from thrifty.utils.logx import EpochLogger
import pickle
import os
import sys
import random
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from dataclasses import dataclass, replace


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h, w, c = obs_dim
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            # nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, act_dim),
            nn.Tanh(),  # squash to [-1,1]
        )

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        return self.model(obs)


class CNNQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h, w, c = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            # nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 + act_dim, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),  # squash to [0,1]
        )

    def forward(self, obs, act):
        obs = obs.permute(0, 3, 1, 2)
        obs = self.conv(obs)
        q = self.linear(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPClassifier(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Sigmoid)
        self.device = device

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.pi(obs).to(self.device)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1], activation, nn.Sigmoid
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLP(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(
            device
        )
        self.pi_safe = MLPClassifier(obs_dim, 1, (128, 128), activation, device).to(
            device
        )
        self.device = device

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

    def classify(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi_safe(obs).cpu().numpy().squeeze()


class Ensemble(nn.Module):
    # Multiple policies
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        num_nets=5,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = nn.Sequential(
            *[
                MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(
                    device
                )
                for _ in range(num_nets)
            ]
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0:  # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0)

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs, act):
        # closer to 1 indicates more safe.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(torch.min(self.q1(obs, act), self.q2(obs, act)).cpu().numpy())


class EnsembleCNN(nn.Module):
    # Multiple policies with image input
    def __init__(self, observation_space, action_space, device, num_nets=5):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [CNNActor(obs_dim, act_dim).to(device) for _ in range(num_nets)]
        self.q1 = CNNQFunction(obs_dim, act_dim).to(device)
        self.q2 = CNNQFunction(obs_dim, act_dim).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            if i >= 0:  # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0).squeeze()

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs, act):
        # closer to 1 indicates more safe.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        if len(act.shape) == 1:
            act = torch.unsqueeze(act, 0)
        with torch.no_grad():
            return float(torch.min(self.q1(obs, act), self.q2(obs, act)).cpu().numpy())


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs], act=self.act_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, obs, act):
        for i in range(len(obs)):
            self.store(obs[i], act[i])

    def save_buffer(self, name="replay"):
        pickle.dump(
            {
                "obs_buf": self.obs_buf,
                "act_buf": self.act_buf,
                "ptr": self.ptr,
                "size": self.size,
            },
            open("{}_buffer.pkl".format(name), "wb"),
        )
        print("buf size", self.size)

    def load_buffer(self, name="replay"):
        p = pickle.load(open("{}_buffer.pkl".format(name), "rb"))
        self.obs_buf = p["obs_buf"]
        self.act_buf = p["act_buf"]
        self.ptr = p["ptr"]
        self.size = p["size"]

    def clear(self):
        self.ptr, self.size = 0, 0


class QReplayBuffer:
    # Replay buffer for training Qrisk
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, pos_fraction=None):
        # # pos_fraction: ensure that this fraction of the batch has rew 1 for better reward propagation
        # if pos_fraction is not None:
        #     pos_size = min(
        #         len(tuple(np.argwhere(self.rew_buf).ravel())),
        #         int(batch_size * pos_fraction),
        #     )
        #     neg_size = batch_size - pos_size
        #     pos_idx = np.array(
        #         random.sample(tuple(np.argwhere(self.rew_buf).ravel()), pos_size)
        #     )
        #     neg_idx = np.array(
        #         random.sample(
        #             tuple(np.argwhere((1 - self.rew_buf)[: self.size]).ravel()),
        #             neg_size,
        #         )
        #     )
        #     idxs = np.hstack((pos_idx, neg_idx))
        #     np.random.shuffle(idxs)
        # else:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def fill_buffer(self, data):
        obs_dim = data["obs"].shape[1]
        act_dim = data["act"].shape[1]
        for i in range(len(data["obs"])):
            if data["done"][i] and not data["rew"][i]:  # time boundary, not really done
                continue
            elif data["done"][i] and data["rew"][i]:  # successful termination
                self.store(
                    data["obs"][i],
                    data["act"][i],
                    np.zeros(obs_dim),
                    data["rew"][i],
                    data["done"][i],
                )
            else:
                self.store(
                    data["obs"][i],
                    data["act"][i],
                    data["obs"][i + 1],
                    data["rew"][i],
                    data["done"][i],
                )

    # def fill_buffer_from_BC(self, data, goals_only=False):
    #     """
    #     Load buffer from offline demos (only obs/act)
    #     goals_only: if True, only store the transitions with positive reward
    #     """
    #     num_bc = len(data["obs"])
    #     obs_dim = data["obs"].shape[1]
    #     for i in range(num_bc - 1):
    #         if data["act"][i][-1] == 1 and data["act"][i + 1][-1] == -1:
    #             # new episode starting
    #             self.store(data["obs"][i], data["act"][i], np.zeros(obs_dim), 1, 1)
    #         elif not goals_only:
    #             self.store(data["obs"][i], data["act"][i], data["obs"][i + 1], 0, 0)
    #     self.store(
    #         data["obs"][num_bc - 1], data["act"][num_bc - 1], np.zeros(obs_dim), 1, 1
    #     )

    def fill_buffer_from_BC(self, data):
        for traj in data:
            num_timesteps = len(traj.acts)
            for timestep in range(num_timesteps):
                if hasattr(traj, "infos"):
                    rew = int(traj.infos[timestep]["success"])
                else:
                    rew = int(timestep == num_timesteps - 1)

                if timestep + 1 == num_timesteps:
                    next_obs = traj.obs[num_timesteps - 1]
                else:
                    next_obs = traj.obs[timestep + 1]

                self.store(
                    traj.obs[timestep],
                    traj.acts[timestep],
                    next_obs,
                    rew=rew,
                    done=timestep == num_timesteps - 1,
                )

    def clear(self):
        self.ptr, self.size = 0, 0


def generate_offline_data(
    env,
    expert_policy,
    num_episodes=0,
    output_file="data.pkl",
    robosuite=False,
    robosuite_cfg=None,
    seed=0,
):
    # Runs expert policy in the environment to collect data
    i, failures = 0, 0
    np.random.seed(seed)
    obs, act, rew = [], [], []
    act_limit = env.action_space.high[0]
    while i < num_episodes:
        print("Episode #{}".format(i))
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        if robosuite:
            robosuite_cfg["INPUT_DEVICE"].start_control()
        while not d:
            a = expert_policy(o)
            if a is None:
                d, r = True, 0
                continue
            a = np.clip(a, -act_limit, act_limit)
            curr_obs.append(o)
            curr_act.append(a)
            o, r, d, _ = env.step(a)
            if robosuite:
                d = (t >= robosuite_cfg["MAX_EP_LEN"]) or env._check_success()
                r = int(env._check_success())
            total_ret += r
            t += 1
        if robosuite:
            if total_ret > 0:  # only count successful episodes
                i += 1
                obs.extend(curr_obs)
                act.extend(curr_act)
            else:
                failures += 1
            env.close()
        else:
            i += 1
            obs.extend(curr_obs)
            act.extend(curr_act)
        print("Collected episode with return {} length {}".format(total_ret, t))
        rew.append(total_ret)
    print("Ep Mean, Std Dev:", np.array(rew).mean(), np.array(rew).std())
    print("Num Successes {} Num Failures {}".format(num_episodes, failures))
    pickle.dump({"obs": np.stack(obs), "act": np.stack(act)}, open(output_file, "wb"))


def thrifty(
    env,
    iters=5,
    actor_critic=Ensemble,
    ac_kwargs=dict(),
    seed=0,
    grad_steps=500,
    obs_per_iter=2000,
    replay_size=int(3e4),
    pi_lr=1e-3,
    batch_size=100,
    logger_kwargs=dict(),
    num_test_episodes=10,
    bc_epochs=5,
    input_file="data.pkl",
    device_idx=0,
    expert_policy=None,
    num_nets=5,
    target_rate=0.01,
    robosuite=False,
    robosuite_cfg=None,
    hg_dagger=None,
    q_learning=False,
    gamma=0.9999,
    init_model=None,
    model_ckpt_path=None,
    input_data=None,
    image_based=False,
):
    """
    obs_per_iter: environment steps per algorithm iteration
    num_nets: number of neural nets in the policy ensemble
    input_file: where initial BC data is stored (output of generate_offline_data())
    target_rate: desired rate of context switching
    robosuite: whether to enable robosuite specific code (and use robosuite_cfg)
    hg_dagger: if not None, use this function as the switching condition (i.e. run HG-DAgger)
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    num_test_episodes: run this many episodes after each iter without interventions
    init_model: initial NN weights
    """
    # logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    del _locals["env"]
    # logger.save_config(_locals)
    if device_idx >= 0:
        device = torch.device("cuda", device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    # if robosuite:
    #     with open(os.path.join(logger_kwargs["output_dir"], "model.xml"), "w") as fh:
    #         fh.write(env.env.sim.model.get_xml())

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    assert act_limit == -1 * env.action_space.low[0], "Action space should be symmetric"
    # horizon = robosuite_cfg["MAX_EP_LEN"]
    horizon = 200

    # initialize actor and classifier NN
    ac = actor_critic(
        env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs
    )
    if init_model:
        ac = torch.load(init_model, map_location=device).to(device)
        ac.device = device
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up optimizers
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)
    print(ac)
    print(ac.pis)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
    )

    # obs = [input_data[i].obs[:-1] for i in range(len(input_data))]
    # acts = [input_data[i].acts for i in range(len(input_data))]
    # obs = np.concatenate(obs)
    # acts = np.concatenate(acts)
    # print(obs.shape, acts.shape)

    # # reshape both
    # obs = obs.reshape(-1, obs.shape[-1])
    # acts = acts.reshape(-1, acts.shape[-1])

    # # shuffle and create small held out set to check valid loss
    # num_bc = len(obs)
    # print(num_bc, obs.shape, acts.shape)
    # idxs = np.arange(num_bc)
    # np.random.shuffle(idxs)
    # replay_buffer.fill_buffer(
    #     obs[idxs][: int(0.9 * num_bc)],
    #     acts[idxs][: int(0.9 * num_bc)],
    # )
    # held_out_data = {
    #     "obs": obs[idxs][int(0.9 * num_bc) :],
    #     "act": acts[idxs][int(0.9 * num_bc) :],
    # }
    # qbuffer = QReplayBuffer(
    #     obs_dim=obs_dim,
    #     act_dim=act_dim,
    #     size=replay_size,
    #     device=device,
    # )
    # qbuffer.fill_buffer_from_BC(input_data)

    # Set up function for computing actor loss
    def compute_loss_pi(data, i):
        o, a = data["obs"], data["act"]
        a_pred = ac.pis[i](o)
        return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))

    def compute_loss_q(data):
        o, a, o2, r, d = (
            data["obs"],
            data["act"],
            data["obs2"],
            data["rew"],
            data["done"],
        )
        # o, a, o2, d = data["obs"], data["act"], data["obs2"], data["done"]
        with torch.no_grad():
            a2 = torch.mean(torch.stack([pi(o2) for pi in ac.pis]), dim=0)
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target Q-values
            q1_t = ac_targ.q1(o2, a2)  # do target policy smoothing?
            q2_t = ac_targ.q2(o2, a2)
            backup = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2

    def update_pi(data, i):
        # run one gradient descent step for pi.
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()

    def update_q(data, timer):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # update targ net
        if timer % 2 == 0:
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(0.995)
                    p_targ.data.add_((1 - 0.995) * p.data)
        return loss_q.item()

    torch.save(ac.state_dict(), model_ckpt_path)
    # train policy
    for i in range(ac.num_nets):
        if ac.num_nets > 1:  # create new datasets via sampling with replacement
            print("Net #{}".format(i))
            # sample random ints
            idxs = np.random.randint(0, len(input_data), size=30)
            # sample 30 random trajectories from the input data to train
            trajectories = [input_data[int(idx)] for idx in idxs]

            if image_based:
                # reshape obs
                for indx, traj in enumerate(trajectories):
                    N, C, H, W = traj.obs.shape
                    trajectories[indx] = replace(traj, obs=traj.obs.reshape(N, H, W, C))

            tmp_buffer = ReplayBuffer(
                obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
            )
            for trajectory in trajectories:
                for timestep in range(len(trajectory.acts)):
                    tmp_buffer.store(
                        trajectory.obs[timestep], trajectory.acts[timestep]
                    )
            # for _ in range(replay_buffer.size):
            #     idx = np.random.randint(replay_buffer.size)
            #     tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            tmp_buffer = replay_buffer

        for j in range(bc_epochs):
            loss_pi = []
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi.append(update_pi(batch, i))
            # validation = []
            # for j in range(len(held_out_data["obs"])):
            #     a_pred = ac.act(held_out_data["obs"][j], i=i)
            #     a_sup = held_out_data["act"][j]
            #     validation.append(sum(a_pred - a_sup) ** 2)
            print("LossPi", sum(loss_pi) / len(loss_pi))
            # print("LossValid", sum(validation) / len(validation))

    # print("training critic")
    # q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    # q_optimizer = Adam(q_params, lr=pi_lr)
    # loss_q = []
    # for _ in range(bc_epochs):
    #     for i in range(grad_steps * 5):
    #         batch = qbuffer.sample_batch(batch_size // 2, pos_fraction=0.1)
    #         loss_q.append(update_q(batch, timer=i))

    print("saving policy")
    torch.save(ac.state_dict(), model_ckpt_path)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append("/scr/aliang80/changepoint_aug/")
    from imitation.data import serialize
    from imitation.data.types import TrajectoryWithRew
    from env_utils import create_single_env
    from metaworld.policies import *

    root_dir = "/scr/aliang80/changepoint_aug"
    env_name = "metaworld-assembly-v2"
    image_based = True
    dataset_file = f"datasets/expert_dataset/image_{image_based}/{env_name}_100_noise_0"
    # load base expert dataset
    print("load dataset from ", dataset_file)
    dataset_file = os.path.join(root_dir, dataset_file)
    full_dataset = serialize.load(dataset_file)
    input_data = full_dataset[:30]
    print("number of expert trajectories: ", len(input_data))

    noise_std = 0
    env = create_single_env(
        env_name, seed=521, image_based=image_based, noise_std=noise_std
    )
    obs, _ = env.reset()

    # run thrifty and save trained policies
    thrifty(
        env,
        actor_critic=EnsembleCNN,
        num_nets=5,
        bc_epochs=25,
        grad_steps=1000,
        batch_size=256,
        pi_lr=3e-4,
        input_data=full_dataset,
        model_ckpt_path=f"/scr/aliang80/changepoint_aug/results/metaworld/thriftydagger_ac_image_{image_based}.pkl",
        image_based=image_based,
    )

    # num_nets = 5
    # ac_kwargs = dict()
    # actor_critic = Ensemble
    # device = "cuda"
    # ac = actor_critic(
    #     env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs
    # )
    # print(ac.q1.q[0].weight.sum())
    # ac.load_state_dict(
    #     torch.load(
    #         "/scr/aliang80/changepoint_aug/results/metaworld/thriftydagger_ac.pkl"
    #     )
    # )
    # print(ac.q1.q[0].weight.sum())

    # risk = []
    # variance = []
    # indx = 40
    # for timestep in range(full_dataset[indx].acts.shape[0]):
    #     obs = torch.from_numpy(full_dataset[indx].obs[timestep])
    #     act = torch.from_numpy(full_dataset[indx].acts[timestep])
    #     print(timestep, " : ", 1 - ac.safety(obs, act), ac.variance(obs))
    #     risk.append(1 - ac.safety(obs, act))
    #     variance.append(ac.variance(obs))
    # sorted_risk = np.argsort(risk)
    # sorted_variance = np.argsort(variance)
    # print(sorted_variance)
