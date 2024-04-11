# -*- coding:utf-8  -*-
import os
import time
import json
import numpy as np
import argparse
import sys

sys.path.append("./olympics_engine")
sys.path.append("env/stock_raw/envs/stock_base_env_cython.cpython-37m-x86_64-linux-gnu.so")

from env.chooseenv import make
from env.utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type


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


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]

            # print("a_obs:", a_obs)
            # print("action_space_list:", action_space_list)
            # print("game.is_act_continuous:", game.is_act_continuous)

            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
    # print(joint_action)
    return joint_action


def set_seed(g, env_name):
    # if env_name.split("-")[0] in ['magent']:
    #     g.reset()
    #     seed = g.create_seed()
    #     g.set_seed(seed)
    print("env_name:", env_name)
    if True:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/agents/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_name,
                 "n_player": g.n_player,
                 "board_height": g.board_height if hasattr(g, "board_height") else None,
                 "board_width": g.board_width if hasattr(g, "board_width") else None,
                 "init_info": g.init_info,
                 "start_time": st,
                 "mode": "terminal",
                 "seed": g.seed if hasattr(g, "seed") else None,
                 "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes
    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        if g.step_cnt % 10 == 0:
            print(step)

        # if render_mode and hasattr(g, "env_core"):
        #     if hasattr(g.env_core, "render"):
        #         g.env_core.render()
        # elif render_mode and hasattr(g, 'render'):
        #     g.render()
        #     print("has_render")
        # if render_mode:
        #     g.render_mode.render()

        '''
            all_observes: [{'observation': {'serverTime': 93000320.0, 'eventTime': 93000430.0, 'code': 2.0, 
            'signal0': 1.3238562962984188, 'signal1': 0.0, 'signal2': 0.0, 'ap0': 4590.8, 'bp0': 4580.335,
            'av0': 3.0, 'bv0': 14.0, 'ap1': 4591.719999999999, 'bp1': 4576.54, 'av1': 3.0, 'bv1': 4.0,
            'ap2': 4591.835, 'bp2': 4576.4710000000005, 'av2': 11.0, 'bv2': 20.0, 'ap3': 4595.400000000001,
            'bp3': 4574.6539999999995, 'av3': 2.0, 'bv3': 11.0, 'ap4': 4597.7, 'bp4': 4574.469999999999,
            'av4': 23.0, 'bv4': 27.0, 'code_net_position': 0, 'ap0_t0': 4590.8}, 
            'new_game': True}]
        
        [{'observation': {'serverTime': 93004406.0, 'eventTime': 93004990.0, 'code': 2.0, 'signal0': -0.5843887380437967, 
        'signal1': -0.69046247746311, 'signal2': -1.693096976594504, 'ap0': 4598.62, 'bp0': 4596.55, 'av0': 8.0,
        'bv0': 3.0, 'ap1': 4600.0, 'bp1': 4594.411, 'av1': 22.0, 'bv1': 4.0, 'ap2': 4604.599999999999,
        'bp2': 4594.365, 'av2': 1.0, 'bv2': 2.0, 'ap3': 4605.658, 'bp3': 4594.3189999999995, 'av3': 7.0, 
        'bv3': 1.0, 'ap4': 4606.900000000001, 'bp4': 4586.269, 'av4': 6.0, 'bv4': 5.0, 'code_net_position': 0, 
        'ap0_t0': 4597.469999999999}, 'new_game': False}]


        '''

        info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
        print("all_observes:", all_observes)
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
        print("joint_act:", joint_act)

        '''
        joint_act: [[[0, 0, 1], array([17.525919], dtype=float32), array([8072.1045], dtype=float32)]]

        joint_act: [[[0, 1, 0], array([24.66666], dtype=float32), array([3948.1428], dtype=float32)]]
        '''

        all_observes, reward, done, info_before, info_after = g.step(joint_act)

        if env_name.split("-")[0] in ["magent"]:
            info_dict["joint_action"] = g.decode(joint_act)
        if info_before:
            info_dict["info_before"] = info_before
        info_dict["reward"] = reward
        if info_after:
            info_dict["info_after"] = info_after
        steps.append(info_dict)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)
    print('n return = ', g.n_return)


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'agents')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":
    env_type = "kafang_stock"
    game = make(env_type, seed=None)

    render_mode = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rule", help="random/rule")
    args = parser.parse_args()

    # policy_list = ["random"] * len(game.agent_nums)
    policy_list = [args.my_ai]  # ["random"] * len(game.agent_nums), here we control agent 2 (green agent)
    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)
