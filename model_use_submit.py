# -*- coding:utf-8  -*-
import os
import time
import json
import numpy as np
import argparse
import sys
from replaybuffer import ReplayBuffer

sys.path.append("./olympics_engine")
sys.path.append("env/stock_raw/envs/stock_base_env_cython.cpython-37m-x86_64-linux-gnu.so")

from env.chooseenv import make
from env.utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type

# from normalization import Normalization
from normalization import Normalization, RewardScaling
from torch.utils.tensorboard import SummaryWriter
from submission import *


if __name__ == "__main__":

    for episode_index in range(4):
        env_type = "kafang_stock"
        game = make(env_type, seed=None)
        all_observes = game.all_observes

        episode_step = 0
        while not game.is_terminal():
            if episode_step % 4000 == 3999:
                print("episode_step:", episode_step)

            action_data = my_controller(all_observes[0], action_space=None)
            action_data = [action_data]
            all_observes, reward, done, info_before, info_after = game.step(action_data)
            episode_step += 1

        print('n_return = ', game.n_return)
