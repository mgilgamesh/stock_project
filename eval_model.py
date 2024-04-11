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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def obs_norm(a_obs):
    # print("a_obs:", a_obs)
    a_obs_data = a_obs
    norm_obs = np.array([
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


def obs_norm_self():
    pass


# 定义：观察量，包含所有的信息
# 动作控制量：0， 1， 2，分别代表买入、什么都不做和卖出
# (1,0,0)是买入,（0，1，0）是什么都不做,(0,0,1)是卖出
#  Volume: [86.37932], Price: [2863.324]
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


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes, agent, state_norm):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    joint_buffer_action = []
    joint_buffer_state = []
    joint_buffer_a_discrete_logprob = []
    joint_buffer_a_continuous_logprob = []
    joint_action_before = []
    joint_reward = []
    safe_all = True
    compute_action = None
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        # print("len(agents_id_list):", len(agents_id_list))

        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]

            # with open('my_file.json', 'w') as json_file:
            #     json.dump(a_obs, json_file)
            # print("a_obs:", a_obs)

            obs_data = obs_norm(a_obs)

            # 观察量归一化，将观察量空间进行压缩，
            # print("obs_data:", obs_data.shape)

            # print("obs_data_after:", obs_data)
            a_discrete, a_continous, a_discrete_logprob, a_continuous_logprob = agent.choose_action(a_obs)

            mm = 5
            # each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            # print("each:", each)
            # print("a_continous_out_data:", a_continous)
            each_new = a_continous.copy()
            # each_new[0] *= 50
            # each_new[1] *= 10000
            # each_new[1] += 0.1

            # print("a_discrete:", a_discrete)
            action_origin = np.zeros(3)
            action_origin[a_discrete[0]] = 1
            # print("action_origin:", list(action_origin))

            each_action = [list(action_origin), float(each_new[0]), float(each_new[1])]
            # 安全判定
            compute_reward, safe = weather_safe(a_obs, each_action)
            safe_all = safe
            # joint_action_before.append(each)
            joint_reward.append(compute_reward)
            joint_action.append(each_action)
            joint_buffer_action.append(a_continous)
            joint_buffer_state.append(obs_data)
            joint_buffer_a_discrete_logprob.append(a_discrete_logprob)
            joint_buffer_a_continuous_logprob.append(a_continuous_logprob)

    # d = joint_action_before, joint_action, joint_buffer_action, joint_buffer_state, joint_buffer_a_discrete_logprob, joint_buffer_a_continuous_logprob
    # print(joint_action)
    # 四个数据：真实控制动作，缓冲池动作，缓冲池状态，缓冲池概率分布
    return joint_action, joint_buffer_action, joint_buffer_state, joint_buffer_a_discrete_logprob, joint_buffer_a_continuous_logprob, joint_reward, safe_all, compute_action


def run_game(env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode, args):
    steps = []

    agent = PPO_continuous(args)
    agent.load()
    state_norm = Normalization(shape=args.state_dim)
    total_steps = 0

    replay_buffer = ReplayBuffer(args)
    episode_num = 0

    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # state_norm = Normalization(shape=args.state_dim)
    writer = SummaryWriter(log_dir='stock/log_train')

    while total_steps < args.max_train_steps:
        # 重置环境
        # print("come_here")
        env_type = "kafang_stock"
        game = make(env_type, seed=None)
        g = game
        # g.reset()
        current_episode_step = 0

        all_observes = g.all_observes
        real_done = False

        if args.use_reward_scaling:
            reward_scaling.reset()

        while not g.is_terminal():
            step = "step%d" % g.step_cnt
            if g.step_cnt % 4000 == 0:
                print("episode_num:", episode_num, "episode_step:", current_episode_step, "total_steps:", total_steps)

            joint_act, joint_buffer_action, joint_buffer_state, joint_buffer_a_discrete_logprob, joint_buffer_a_continuous_logprob, joint_reward, safe_all, compute_action = get_joint_action_eval(
                g,
                multi_part_agent_ids,
                policy_list,
                actions_spaces,
                all_observes,
                agent,
                state_norm)
            # joint_act = [[[0.0, 0.0, 1.0], 88.844604, 7403.3276]]
            real_action = joint_act

            # max:100,10000
            # print("joint_act:", real_action)
            action_discrete = real_action[0][0].index(1)
            action_continous = np.array((float(real_action[0][1]), float(real_action[0][2])))
            # print("real_action:", real_action)
            # print("compute_action:", compute_action)

            #
            # print("real_action:", real_action)
            # print("real_action_model:", real_action_model)
            # print("*****************")
            code_pnl_before = all_observes[0]["observation"]["code_pnl"]
            all_observes, reward, done, info_before, info_after = g.step(real_action)

            current_episode_step += 1
            total_steps += 1

            # 回合结束时候，
            if g.is_terminal():
                # if episode_num % 30 == 0:
                print('n_return = ', g.n_return)
                if g.n_return[0] > 0:
                    print("good")

            code_pnl_after = all_observes[0]["observation"]["code_pnl"]
            writer.add_scalar('code_pnl:{}'.format(env_name), code_pnl_after, global_step=total_steps)
            delta_code_pnl = code_pnl_after - code_pnl_before

            # 没有下一个obs了，说明结束了
            if all_observes[0]['observation'] is None:
                dw = True
                print("dw:", dw)

            # done等于0表示结束
            if done == 0:
                pass


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'agents')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":
    env_type = "kafang_stock"
    game = make(env_type, seed=None)
    # game.reset()
    render_mode = True
    # parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--my_ai", default="random", help="random/rule")
    parser.add_argument("--max_train_steps", type=int, default=int(4e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=2048,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=20, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=220,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gpu", type=bool, default=True, help="使用gpu")

    args = parser.parse_args()
    # 自定义参数
    args.max_action = 5000
    args.state_dim = 25

    args.continuous_action_dim = 2
    args.discrete_action_dim = 3

    args.max_episode_steps = 200

    # policy_list = ["random"] * len(game.agent_nums)
    policy_list = [args.my_ai]  # ["random"] * len(game.agent_nums), here we control agent 2 (green agent)
    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    run_game(env_type, multi_part_agent_ids, actions_space, policy_list, render_mode, args)
