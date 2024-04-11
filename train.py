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
from ppo_discrete import PPO_discrete
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


def obs_norm_self():
    pass


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes, agent, state_norm):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    joint_buffer_action = []
    joint_buffer_state = []
    joint_buffer_a_discrete_logprob = []

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
            obs_data = obs_norm(a_obs)
            a_discrete, a_discrete_logprob = agent.choose_action(a_obs)
            joint_buffer_action.append(a_discrete)
            joint_buffer_state.append(obs_data)
            joint_buffer_a_discrete_logprob.append(a_discrete_logprob)

    return joint_buffer_action, joint_buffer_state, joint_buffer_a_discrete_logprob


def run_game(env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode, args):
    steps = []

    agent = PPO_discrete(args)
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
        current_episode_step = 0

        all_observes = g.all_observes

        if args.use_reward_scaling:
            reward_scaling.reset()

        while not g.is_terminal():
            step = "step%d" % g.step_cnt
            if g.step_cnt % 4000 == 0:
                print("episode_num:", episode_num, "episode_step:", current_episode_step, "total_steps:", total_steps)

            joint_buffer_action, joint_buffer_state, joint_buffer_a_discrete_logprob = get_joint_action_eval(
                g,
                multi_part_agent_ids,
                policy_list,
                actions_spaces,
                all_observes,
                agent,
                state_norm)

            if joint_buffer_action[0] == 0:
                volumn_0 = min(all_observes[0]["observation"]['av0'],
                               300 - all_observes[0]["observation"]['code_net_position'])
                price_0 = all_observes[0]["observation"]['ap0']

                joint_act = [[[1.0, 0.0, 0.0], float(volumn_0), float(price_0)]]
            if joint_buffer_action[0] == 1:
                joint_act = [[[0.0, 1.0, 0.0], all_observes[0]["observation"]['av0'], all_observes[0]["observation"]['ap0']]]
            if joint_buffer_action[0] == 2:
                volumn_2 = min(all_observes[0]["observation"]['bv0'],
                               300 + all_observes[0]["observation"]['code_net_position'])
                price_2 = all_observes[0]["observation"]['bp0']
                joint_act = [[[0.0, 0.0, 1.0], float(volumn_2), float(price_2)]]

            # print("joint_act:", joint_act)

            code_pnl_before = all_observes[0]["observation"]["code_pnl"]
            all_observes, reward, done, info_before, info_after = g.step(joint_act)

            # print("reward:", reward)

            current_episode_step += 1
            total_steps += 1
            dw = False
            # 将连续动作和离散动作分别存储，一起训练，具体的训练细节可以参考chatgpt。

            buffer_action_discrete = joint_buffer_action[0]

            buffer_state = np.array(joint_buffer_state).squeeze()
            buffer_action_discrete_logprob = np.array(joint_buffer_a_discrete_logprob).squeeze()
            real_done = False

            # 回合结束时候，
            if g.is_terminal():

                code_pnl_after = -5000
                delta_code_pnl = code_pnl_after - code_pnl_before
                compute_reward = delta_code_pnl / 100
                buffer_reward = np.array(compute_reward)
                episode_num += 1
                real_done = True
                dw = True
                buffer_done = real_done
                buffer_next_state = np.zeros(31)
                buffer_dw = dw
                # store(self, s, a, a_logprob, r, s_, dw, done):
                replay_buffer.store(buffer_state, buffer_action_discrete, buffer_action_discrete_logprob, buffer_reward,
                                    buffer_next_state, buffer_dw, buffer_done
                                    )

                # if episode_num % 30 == 0:
                print('n_return = ', g.n_return)
                if g.n_return[0] > 0:
                    print("good")

                # 回合结束时更新
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0
                break

            # if current_episode_step == args.max_episode_steps:
            #     real_done = True
            code_pnl_after = all_observes[0]["observation"]["code_pnl"]
            writer.add_scalar('code_pnl:{}'.format(env_name), code_pnl_after, global_step=total_steps)
            delta_code_pnl = code_pnl_after - code_pnl_before
            compute_reward = delta_code_pnl / 100

            buffer_reward = np.array(compute_reward)
            buffer_done = real_done
            buffer_next_state = np.array(obs_norm(all_observes[0])).squeeze()
            buffer_dw = dw

            replay_buffer.store(buffer_state, buffer_action_discrete, buffer_action_discrete_logprob, buffer_reward,
                                buffer_next_state, buffer_dw, buffer_done
                                )

            if total_steps % 200000 == 0:
                agent.save()
                agent.save_all()

            # 回合未结束时更新
            if replay_buffer.count == args.batch_size:
                # print("update")
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0


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
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=320,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
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
    args.state_dim = 31

    args.continuous_action_dim = 2
    args.action_dim = 3

    args.max_episode_steps = 200

    # policy_list = ["random"] * len(game.agent_nums)
    policy_list = [args.my_ai]  # ["random"] * len(game.agent_nums), here we control agent 2 (green agent)
    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    run_game(env_type, multi_part_agent_ids, actions_space, policy_list, render_mode, args)
