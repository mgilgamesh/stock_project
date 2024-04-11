import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.distributions import Categorical
import argparse
import numpy as np


def obs_norm(a_obs):
    # print("a_obs:", a_obs)
    a_obs_data = a_obs
    norm_obs = np.array([
        a_obs_data["observation"]["signal0"] / 2,
        a_obs_data["observation"]["signal1"] / 2,
        a_obs_data["observation"]["signal2"] / 2,

        a_obs_data["observation"]["signal0"] / 2,
        a_obs_data["observation"]["signal1"] / 2,
        a_obs_data["observation"]["signal2"] / 2,

        a_obs_data["observation"]["signal0"] / 2,
        a_obs_data["observation"]["signal1"] / 2,
        a_obs_data["observation"]["signal2"] / 2,

        a_obs_data["observation"]["ap0"] / 10000,
        a_obs_data["observation"]["bp0"] / 10000,
        a_obs_data["observation"]["av0"] / 100,
        a_obs_data["observation"]["bv0"] / 100,

        a_obs_data["observation"]["ap1"] / 10000,
        a_obs_data["observation"]["bp1"] / 10000,
        a_obs_data["observation"]["av1"] / 100,
        a_obs_data["observation"]["bv1"] / 100,

        a_obs_data["observation"]["ap2"] / 10000,
        a_obs_data["observation"]["bp2"] / 10000,
        a_obs_data["observation"]["av2"] / 100,
        a_obs_data["observation"]["bv2"] / 100,

        a_obs_data["observation"]["ap3"] / 10000,
        a_obs_data["observation"]["bp3"] / 10000,
        a_obs_data["observation"]["av3"] / 100,
        a_obs_data["observation"]["bv3"] / 100,

        a_obs_data["observation"]["ap4"] / 10000,
        a_obs_data["observation"]["bp4"] / 10000,
        a_obs_data["observation"]["av4"] / 100,
        a_obs_data["observation"]["bv4"] / 100,

        int(a_obs_data["observation"]["code_net_position"]) / 1000,
        a_obs_data["observation"]["ap0_t0"] / 10000
    ])
    return norm_obs

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.dropout(s)
        s = self.activate_func(self.fc2(s))
        s = self.dropout(s)
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob




parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")

args = parser.parse_args()
# 自定义参数
args.max_action = 5000
args.state_dim = 31

args.action_dim = 3
args.max_episode_steps = 200
args.hidden_width = 320
args.use_tanh = True
args.use_orthogonal_init = True

device = torch.device("cpu")
file_path = "upload_files_0229/202402290830_world_model/actor.pkl"
# file_path = "/home/kf/notebook/code/workspace/Competition_RL4Stock/agents/backtest202402212351/actor.pkl"
actor = Actor(args).to(device)

actor.load_state_dict(torch.load(file_path))

actor.eval()


def my_controller(observation, action_space, is_act_continuous=False):
    ap0 = observation["observation"]["ap0"]
    ap1 = observation["observation"]["ap1"]
    ap2 = observation["observation"]["ap2"]
    ap3 = observation["observation"]["ap3"]
    ap4 = observation["observation"]["ap4"]

    av0 = observation["observation"]["av0"]
    av1 = observation["observation"]["av1"]
    av2 = observation["observation"]["av2"]
    av3 = observation["observation"]["av3"]
    av4 = observation["observation"]["av4"]

    bp0 = observation["observation"]["bp0"]
    bp1 = observation["observation"]["bp1"]
    bp2 = observation["observation"]["bp2"]
    bp3 = observation["observation"]["bp3"]
    bp4 = observation["observation"]["bp4"]

    bv0 = observation["observation"]["bv0"]
    bv1 = observation["observation"]["bv1"]
    bv2 = observation["observation"]["bv2"]
    bv3 = observation["observation"]["bv3"]
    bv4 = observation["observation"]["bv4"]
    code_net_position = observation["observation"]["code_net_position"]

    s = obs_norm(observation)
    s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
    # dist = Categorical(probs=actor(s))

    a_prob = actor(s).detach().numpy().flatten()
    action = np.argmax(a_prob)
    # print("action:", action)
	
    if action == 0:
        volumn_0 = min(observation["observation"]['av0'],
                       300 - observation["observation"]['code_net_position'])
        price_0 = observation["observation"]['ap0']

        joint_act = [[1.0, 0.0, 0.0], [float(volumn_0)], [float(price_0)]]
    if action == 1:
        joint_act = [[0.0, 1.0, 0.0], [observation["observation"]['av0']], [observation["observation"]['ap0']]]
    if action == 2:
        volumn_2 = min(observation["observation"]['bv0'],
                       300 + observation["observation"]['code_net_position'])
        price_2 = observation["observation"]['bp0']
        joint_act = [[0.0, 0.0, 1.0], [float(volumn_2)], [float(price_2)]]

    # print("joint_act:",joint_act)
    # [[0, 1, 0], [0], [0]]
    # [[1.0, 0.0, 0.0], [0.0], [2957.938]]
    # each_action = [[list(action_origin), float(each_new[0]), float(each_new[1])]]

    return joint_act
