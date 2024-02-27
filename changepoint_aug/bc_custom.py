import logging

logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable

# from r3meval.utils.logger import DataLog
from tqdm import tqdm
from pathlib import *
from imitation.data import serialize
import gc
from env_utils import create_single_env


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def do_rollout(
    num_traj,
    env,
    policy,
    eval_mode=False,
    horizon=1e6,
    base_seed=None,
    env_kwargs=None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """
    if base_seed is not None:
        try:
            env.set_seed(base_seed)
        except:
            env.seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    # horizon = min(horizon, env.horizon)
    paths = []

    ep = 0
    while ep < num_traj:
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            try:
                env.set_seed(seed)
            except:
                env.seed(seed)
            np.random.seed(seed)

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o, env_info_step = env.reset()
        done = False
        t = 0
        ims = []
        try:
            ims.append(env.env.env.get_image())
        except:
            ## For state based learning
            pass

        ## MetaWorld vs. Adroit/Kitchen syntax
        try:
            init_state = env.__getstate__()
        except:
            init_state = env.get_env_state()

        while t < horizon and done != True:
            a = policy(torch.from_numpy(o).cuda().float()).cpu().detach().numpy()
            next_o, r, done, truncated, env_info_step = env.step(a)
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            env_infos.append(env_info_step)
            try:
                ims.append(env.env.env.get_image())
            except:
                pass
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminated=done,
            init_state=init_state,
            env_infos=env_infos,
            images=ims,
        )

        paths.append(path)
        ep += 1

    del env
    gc.collect()
    return paths


class BC:
    def __init__(
        self,
        root_dir,
        dataset_file,
        policy,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        optimizer=None,
        loss_type="MSE",  # can be 'MLE' or 'MSE'
        save_logs=True,
        set_transforms=False,
        finetune=False,
        proprio=1,
        encoder_params=[],
        **kwargs,
    ):
        self.policy = policy
        self.epochs = epochs
        self.mb_size = batch_size
        # self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.finetune = finetune
        self.proprio = proprio
        self.steps = 0

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(2052, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),
        )
        self.policy.cuda()

        # construct optimizer
        self.optimizer = (
            torch.optim.Adam(
                list(self.policy.parameters()) + list(encoder_params), lr=lr
            )
            if optimizer is None
            else optimizer
        )

        # Loss criterion if required
        # if loss_type == "MSE":
        self.loss_criterion = torch.nn.MSELoss()

        # load dataset
        print("load base expert dataset from ", dataset_file)
        dataset_file = Path(root_dir) / dataset_file
        expert_trajectories = serialize.load(dataset_file)
        self.dataset = expert_trajectories
        print(
            "number of expert trajectories in full dataset: ", len(expert_trajectories)
        )

        # # make logger
        # if self.save_logs:
        #     self.logger = DataLog()

    def loss(self, data, idx=None):
        if self.loss_type == "MLE":
            return self.mle_loss(data, idx)
        elif self.loss_type == "MSE":
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data["observations"].shape[0]) if idx is None else idx
        if type(data["observations"]) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data["observations"][idx]
        act = data["expert_actions"][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None):
        idx = range(data["observations"].shape[0]) if idx is None else idx
        if type(data["observations"]) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data["observations"][idx]
        ## Encode images with environments encode function
        # obs = self.encodefn(obs, finetune=self.finetune)
        act_expert = torch.from_numpy(data["expert_actions"][idx]).cuda().float()
        if type(obs) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False).cuda()

        ## Concatenate proprioceptive data
        if self.proprio:
            proprio = data["proprio"][idx]
            if type(proprio) is not torch.Tensor:
                proprio = Variable(
                    torch.from_numpy(proprio).float(), requires_grad=False
                ).cuda()
            obs = torch.cat([obs, proprio], -1)
        if type(act_expert) is not torch.Tensor:
            act_expert = Variable(
                torch.from_numpy(act_expert).float(), requires_grad=False
            )
        act_pi = self.policy(obs)
        return self.loss_criterion(act_pi, act_expert.detach())

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all(
            [k in data.keys() for k in ["observations", "expert_actions"]]
        )
        assert validate_keys is True
        ts = timer.time()
        num_samples = data["observations"].shape[0]

        # log stats before
        # if self.save_logs:
        #     loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
        #     self.logger.log_kv("loss_before", loss_val)

        # train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            if ep % 10 == 0:
                env = create_single_env(
                    "metaworld-assembly-v2",
                    seed=0,
                    image_based=True,
                    noise_std=0.0,
                    freeze_rand_vec=True,
                    use_pretrained_img_embeddings=True,
                    add_proprio=True,
                    embedding_name="resnet50",
                    history_window=1,
                )
                paths = do_rollout(
                    5,
                    env,
                    self.policy,
                    eval_mode=True,
                    horizon=200,
                    base_seed=0,
                    env_kwargs=None,
                )
                # compute success rate
                successes = [path["env_infos"][-1]["success"] for path in paths]
                success_rate = sum(successes) / len(successes)
                print(f"Epoch: {ep} | Success rate: {success_rate}")
                # torch.save(self.policy, "policy.pt")

            epoch_loss = 0.0
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss = self.loss(data, idx=rand_idx)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                self.steps += 1
            epoch_loss /= int(num_samples / self.mb_size)
            print(f"Epoch {ep} Loss: {epoch_loss}")

        # params_after_opt = self.policy.get_param_values()
        # self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        # log stats after
        # if self.save_logs:
        #     self.logger.log_kv("epoch", self.epochs)
        #     loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
        #     self.logger.log_kv("loss_after", loss_val)
        #     self.logger.log_kv("time", (timer.time() - ts))

    def train(self, pixel=True, **kwargs):
        ## If using proprioception, select the first N elements from the state observation
        ## Assumes proprioceptive features are at the front of the state observation
        proprio = np.concatenate([path.obs for path in self.dataset])
        proprio = proprio[:, -self.proprio :]
        observations = np.concatenate(
            [traj.obs[:-1, : -self.proprio] for traj in self.dataset]
        )
        ## Extract actions
        expert_actions = np.concatenate([path.acts for path in self.dataset])
        data = dict(
            observations=observations, proprio=proprio, expert_actions=expert_actions
        )
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)


def main():
    bc = BC(
        root_dir="/scr/aliang80/changepoint_aug",
        dataset_file="datasets/expert_dataset/image_True_pretrained_True_r3m_resnet50_add_proprio_stack_1/metaworld-assembly-v2_100_noise_0",
        policy=None,
        epochs=1000,
        batch_size=64,
        lr=1e-3,
        optimizer=None,
        loss_type="MSE",
        save_logs=True,
        set_transforms=False,
        finetune=False,
        proprio=4,
        encoder_params=[],
    )
    bc.train()


if __name__ == "__main__":
    main()
