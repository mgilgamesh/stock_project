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
from ppo_continuous import PPO_continuous
# from normalization import Normalization
from normalization import Normalization, RewardScaling
from torch.utils.tensorboard import SummaryWriter
from submission import *
from ppo_continuous import PPO_continuous


def weather_safe(a_obs, each_action):
    # print("a_obs:", a_obs)
    # print("each_action:", each_action)
    # obs数据重新组合
    ap0 = a_obs["observation"]["ap0"]
    ap1 = a_obs["observation"]["ap1"]
    ap2 = a_obs["observation"]["ap2"]
    ap3 = a_obs["observation"]["ap3"]
    ap4 = a_obs["observation"]["ap4"]

    av0 = a_obs["observation"]["av0"]
    av1 = a_obs["observation"]["av1"]
    av2 = a_obs["observation"]["av2"]
    av3 = a_obs["observation"]["av3"]
    av4 = a_obs["observation"]["av4"]

    bp0 = a_obs["observation"]["bp0"]
    bp1 = a_obs["observation"]["bp1"]
    bp2 = a_obs["observation"]["bp2"]
    bp3 = a_obs["observation"]["bp3"]
    bp4 = a_obs["observation"]["bp4"]

    bv0 = a_obs["observation"]["bv0"]
    bv1 = a_obs["observation"]["bv1"]
    bv2 = a_obs["observation"]["bv2"]
    bv3 = a_obs["observation"]["bv3"]
    bv4 = a_obs["observation"]["bv4"]

    code_net_position = a_obs["observation"]["code_net_position"]

    Volume = each_action[1]
    Price = each_action[2]
    order_type = np.argmax(each_action[0])
    # print("Volume:", Volume, "Price:", Price, "order_type:", order_type)
    real_reward = 0
    # 危险情况重写
    # 第一个volume小于0
    safe = True
    if Volume < 0:
        real_reward += -20
        safe = False
    # 第二个volume=16>市场总量
    if order_type == 0 and Volume > (av0 + av1 + av2 + av3 + av4):
        real_reward += -20
        safe = False
    # 第三个volume=13>市场总量（1+1+4+3+3=12）
    if order_type == 2 and Volume > (bv0 + bv1 + bv2 + bv3 + bv4):
        real_reward += -20
        safe = False
    # 第四个code_net_position + volume > 300,超过环境设定的持仓最高300的条件。
    if order_type == 0 and code_net_position + Volume > 300:
        real_reward += -20
        safe = False
    # price < askPx1,但volume >= 0
    if order_type == 0 and Price <= ap0 and Volume >= 0:
        real_reward += -20
        safe = False
    # askPx1 < price < askPx2，但volume > askVlm1
    if order_type == 0 and Price < ap1 and Price > ap0 and Volume > av0:
        real_reward += -20
        safe = False
    # askPx2 < price < askPx3，但volume > (askVlm1 + askVlm2)
    if order_type == 0 and Price < ap2 and Price > ap1 and Volume > (av0 + av1):
        real_reward += -20
        safe = False
    # askPx3 < price < askPx4，但volume > (askVlm1 + askVlm2 + askVlm3)
    if order_type == 0 and Price < ap3 and Price > ap2 and Volume > (av0 + av1 + av2):
        real_reward += -20
        safe = False
    # askPx4 < price < askPx5，但volume > (askVlm1 + askVlm2 + askVlm3 + askVlm4)
    if order_type == 0 and Price < ap4 and Price > ap3 and Volume > (av0 + av1 + av2 + av3):
        real_reward += -20
        safe = False
    # askPx5 < price，但volume > (askVlm1 + askVlm2 + askVlm3 + askVlm4 + askVlm5)
    if order_type == 0 and Price > ap4 and Volume > (av0 + av1 + av2 + av3 + av4):
        real_reward += -20
        safe = False
    # code_net_position - volume < -300,低于环境设定的持仓低于-300的条件。
    if order_type == 2 and code_net_position - Volume < -300:
        real_reward += -20
        safe = False
    # price > bidPx1
    if order_type == 2 and Price > bp0:
        real_reward += -20
        safe = False
    # bidPx1 > price > bi
    # dPx2,但volume > bidVlm1
    if order_type == 2 and bp0 > Price and Price > bp1 and Volume > bv0:
        real_reward += -20
        safe = False
    # bidPx2 > price > bidPx3,但volume > bidVlm1 + bidVlm2
    if order_type == 2 and bp1 > Price and Price > bp2 and Volume > (bv0 + bv1):
        real_reward += -20
        safe = False
    # bidPx3 > price > bidPx4,但volume > bidVlm1 + bidVlm2 + bidVlm3
    if order_type == 2 and bp2 > Price and Price > bp3 and Volume > (bv0 + bv1 + bv2):
        real_reward += -20
        safe = False
    # bidPx4 > price > bidPx5,但volume > bidVlm1 + bidVlm2 + bidVlm3 + bidVlm4
    if order_type == 2 and bp3 > Price and Price > bp4 and Volume > (bv0 + bv1 + bv2 + bv3):
        real_reward += -20
        safe = False
    # bidPx5 > price,但volume > bidVlm1 + bidVlm2 + bidVlm3 + bidVlm4 + bidVlm5
    if order_type == 2 and bp4 > Price and Volume > (bv0 + bv1 + bv2 + bv3 + bv4):
        real_reward += -20
        safe = False
    # print("a_obs:", a_obs)
    return real_reward, safe


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")

    args = parser.parse_args()
    # 自定义参数
    args.max_action = 5000
    args.state_dim = 25

    args.continuous_action_dim = 2
    args.discrete_action_dim = 3

    args.max_episode_steps = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actor = torch.load('model_all/actor_all.pkl')

    for episode_index in range(4):
        env_type = "kafang_stock"
        game = make(env_type, seed=None)
        all_observes = game.all_observes

        episode_step = 0

        while not game.is_terminal():
            if episode_step % 4000 == 3999:
                print("episode_step:", episode_step)
            real_obs = all_observes[0]
            s = obs_norm(real_obs)
            # 根据real_obs来

            ap0 = real_obs["observation"]["ap0"]
            ap1 = real_obs["observation"]["ap1"]
            ap2 = real_obs["observation"]["ap2"]
            ap3 = real_obs["observation"]["ap3"]
            ap4 = real_obs["observation"]["ap4"]

            av0 = real_obs["observation"]["av0"]
            av1 = real_obs["observation"]["av1"]
            av2 = real_obs["observation"]["av2"]
            av3 = real_obs["observation"]["av3"]
            av4 = real_obs["observation"]["av4"]

            bp0 = real_obs["observation"]["bp0"]
            bp1 = real_obs["observation"]["bp1"]
            bp2 = real_obs["observation"]["bp2"]
            bp3 = real_obs["observation"]["bp3"]
            bp4 = real_obs["observation"]["bp4"]

            bv0 = real_obs["observation"]["bv0"]
            bv1 = real_obs["observation"]["bv1"]
            bv2 = real_obs["observation"]["bv2"]
            bv3 = real_obs["observation"]["bv3"]
            bv4 = real_obs["observation"]["bv4"]
            code_net_position = real_obs["observation"]["code_net_position"]

            # print("s:", s)
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
            with torch.no_grad():
                dist_discrete = Categorical(probs=actor(s)[1])
                a_discrete = dist_discrete.sample()
                a_logprob_discrete = dist_discrete.log_prob(a_discrete)
                a_type = int(a_discrete)
                dist_continous = actor.get_dist(s)
                a_continous = dist_continous.sample()  # Sample the action according to the probability distribution
                a_continous = torch.clamp(a_continous, 0.0001, 1)  # [-max,max]
                # 0 是量，1是价格
                a_continous[:, 0] *= 50
                a_continous[:, 1] *= 10000
                if a_type == 0:
                    max_volumn_1 = av0 + av1 + av2 + av3 + av4
                    # print("max_volumn_1:", max_volumn_1)
                    max_volumn_2 = 300 - code_net_position
                    # print("max_volumn_2:", max_volumn_2)
                    min_price_1 = ap0
                    # print("min_price_1:", min_price_1)
                    max_volumn_3 = 1000
                    max_volumn_3_2 = 1000
                    # print("a_continous[:, 1]:", a_continous[:, 1])

                    if float(a_continous[:, 1]) < ap0:
                        max_volumn_3_2 = av0
                    if ap0 < float(a_continous[:, 1]) < ap1:
                        max_volumn_3 = av0
                    if ap1 < float(a_continous[:, 1]) < ap2:
                        max_volumn_3 = av0 + av1
                    if ap2 < float(a_continous[:, 1]) < ap3:
                        max_volumn_3 = av0 + av1 + av2
                    if ap3 < float(a_continous[:, 1]) < ap4:
                        max_volumn_3 = av0 + av1 + av2 + av3
                    if ap4 < float(a_continous[:, 1]):
                        max_volumn_3 = av0 + av1 + av2 + av3 + av4

                    all_max_volumn = min(max_volumn_1, max_volumn_2, max_volumn_3, max_volumn_3_2)
                    # print("all_max_volumn:", all_max_volumn)
                    a_continous[:, 0] = torch.clamp(a_continous[:, 0], 0.01, all_max_volumn - 0.00001)
                    a_continous[:, 1] = torch.clamp(a_continous[:, 1], min_price_1, 15000)
                    if round(code_net_position, 0) == 300:
                        a_continous[:, 0] = 0
                    a_continous[:, 1] = a_continous[:, 1] + 0.01

                    # print("out_data:", a_continous)
                if a_type == 1:
                    pass

                if a_type == 2:
                    max_volumn_1 = bv0 + bv1 + bv2 + bv3 + bv4
                    max_volumn_2 = 300 + code_net_position
                    max_volumn_3_2 = 1000
                    max_volumn_3_3 = 1000

                    max_price_1 = bp0
                    # print("max_price_1:", max_price_1)
                    max_volumn_3 = 1000

                    if float(a_continous[:, 1]) > bp0:
                        max_volumn_3_2 = bv0
                    if bp0 > float(a_continous[:, 1]) > bp1:
                        max_volumn_3 = bv0
                    if bp1 > float(a_continous[:, 1]) > bp2:
                        max_volumn_3 = bv0 + bv1
                    if bp2 > float(a_continous[:, 1]) > bp3:
                        max_volumn_3 = bv0 + bv1 + bv2
                    if bp3 > float(a_continous[:, 1]) > bp4:
                        max_volumn_3 = bv0 + bv1 + bv2 + bv3
                    if float(a_continous[:, 1]) > bp4:
                        max_volumn_3 = bv0 + bv1 + bv2 + bv3 + bv4
                    if float(a_continous[:, 1]) > min(bp0, bp1, bp2, bp3, bp4):
                        max_volumn_3_3 = bp4

                    all_max_volumn = min(max_volumn_1, max_volumn_2, max_volumn_3, max_volumn_3_2,
                                             max_volumn_3_3)
                    a_continous[:, 0] = torch.clamp(a_continous[:, 0], 0.01, all_max_volumn)
                    a_continous[:, 1] = torch.clamp(a_continous[:, 1], 0.01, max_price_1 - 0.5)

                    if round(code_net_position - float(a_continous[:, 0]), 0) == -300:
                        a_continous[:, 0] -= 0.01
                        if a_continous[:, 0] < 0:
                            a_continous[:, 0] = 0
                dis_action = a_discrete.cpu().numpy().flatten()
                con_action = a_continous.cpu().numpy().flatten()

                action_origin = np.zeros(3)
                action_origin[dis_action[0]] = 1
                # print("action_origin:", list(action_origin))

                each_action = [list(action_origin), float(con_action[0]), float(con_action[1])]

                # print("each_action_here:", each_action)
                _, safe = weather_safe(real_obs, each_action)
                if not safe:
                    print("no_safe")
                    if dis_action[0] == 0:
                        con_action[0] = av0
                        con_action[1] = ap0
                    if dis_action[0] == 2:
                        con_action[0] = bv0
                        con_action[1] = bp0

                a_discrete = dis_action
                a_continous = con_action
                each_new = a_continous.copy()
                action_origin = np.zeros(3)
                action_origin[a_discrete[0]] = 1
                each_action = [list(action_origin), float(each_new[0]), float(each_new[1])]
                # print("each_action:", each_action)
                all_observes, reward, done, info_before, info_after = game.step([each_action])
            episode_step += 1

        print('n_return = ', game.n_return)
