"""対局を指定した回数行うスクリプト．対局結果の出力も可能．
"""

import argparse
import json
import os
import random
from datetime import datetime

import custom_evaluate
import mjx
import mjx.agents
import torch
from client.agent import CustomAgentBase
from custom_ai import Net, Net2
from mjx.const import ActionType
from server import convert_log
from torch import Tensor, nn, optim, utils

# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．


class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        # self.model = torch.load('models/second_model.pth')
        self.model = Net2()
        self.model.load_state_dict(torch.load("models/fifth_model.pth"))

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        # // 0~33: Discard m1~rd
        # // 34,35,36: Discard m5(red), p5(red), s5(red)
        # // 37~70: Tsumogiri m1~rd
        # // 71,72,73: Tsumogiri m5(red), p5(red), s5(red)
        # // 74~94: Chi m1m2m3 ~ s7s8s9
        # // 95,96,97: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
        # // 98,99,100: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
        # // 101,102,103: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
        # // 104~137: Pon m1~rd
        # // 138,139,140: Pon m5(w/ red), s5(w/ red), p5(w/ red)
        # // 141~174: Kan m1~rd
        # // 175: Tsumo
        # // 176: Ron
        # // 177: Riichi
        # // 178: Kyuushu
        # // 179: No
        # // 180: Dummy

        # def find_action(legal_action_idx):
        #     legal_actions_idx = [element.to_idx() for element in obs.legal_actions()]
        #     if legal_action_idx in legal_actions_idx:
        #         action_index = legal_actions_idx.index(legal_action_idx)
        #     return obs.legal_actions()[action_index]

        # return_action = find_action(175)
        # legal_actions = obs.legal_actions()
        # legal_actions_idx = [element.to_idx() for element in obs.legal_actions()]
        # print(obs.action_mask())
        # print(legal_actions_idx)

        # if 175 in legal_actions_idx:  # Tsumo
        #     action_index = legal_actions_idx.index(175)
        #     print("ツモ")
        #     print(obs.legal_actions()[action_index].type())
        #     return obs.legal_actions()[action_index]
        # elif 176 in legal_actions_idx:  # Ron
        #     action_index = legal_actions_idx.index(176)
        #     print("ロン")
        #     print(obs.legal_actions()[action_index].type())
        #     return obs.legal_actions()[action_index]
        # elif 177 in legal_actions_idx:  # Riichi
        #     action_index = legal_actions_idx.index(177)
        #     print("リーチ")
        #     print(obs.legal_actions()[action_index].type())
        #     return obs.legal_actions()[action_index]
        # elif not set(legal_actions_idx).isdisjoint(set(range(74, 175))):
        #     legal_actions = [element for element in legal_actions if element.to_idx() not in range(74, 175)]
        #     return_action = random.choice(legal_actions)
        #     # print("除外")
        #     # print(return_action.type())
        #     return return_action
        # elif not set(legal_actions_idx).isdisjoint(set(range(0, 74))):
        #     legal_actions = [element for element in legal_actions if element.to_idx() in range(0, 74)]
        #     effective_discard_types = obs.curr_hand().effective_discard_types()
        #     effective_discards = [
        #         a for a in legal_actions if a.tile().type() in effective_discard_types
        #     ]
        #     return_action = random.choice(effective_discards)
        #     # print("除外")
        #     # print(return_action.type())
        #     return return_action

        # print(obs.legal_actions()[0].to_json())
        # print(obs.legal_actions())
        # print([element.to_json() for element in obs.legal_actions()])
        # print([element.tile().type() for element in obs.legal_actions()])
        # print([element.tile().id() for element in obs.legal_actions()])
        # print([element.type() for element in obs.legal_actions()])
        # print([element.to_idx() for element in obs.legal_actions()])
        # # print([element.to_proto() for element in obs.legal_actions()])
        # # print(obs.to_features("mjx-large-v0"))
        # # print(obs.MjxLargeV0().current_hand(obs))
        # print()
        # print()
        # legal_actions = observation.legal_actions()
        # if len(legal_actions) == 1:
        #     return legal_actions[0]

        # # if it can win, just win
        # win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        # if len(win_actions) >= 1:
        #     assert len(win_actions) == 1
        #     return win_actions[0]

        # # if it can declare riichi, just declar riichi
        # riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        # if len(riichi_actions) >= 1:
        #     assert len(riichi_actions) == 1
        #     return riichi_actions[0]

        # # if it can apply chi/pon/open-kan, choose randomly
        # steal_actions = [
        #     a
        #     for a in legal_actions
        #     if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        # ]
        # if len(steal_actions) >= 1:
        #     return random.choice(steal_actions)

        # # if it can apply closed-kan/added-kan, choose randomly
        # kan_actions = [
        #     a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN, ActionType.ADDED_KAN]
        # ]
        # if len(kan_actions) >= 1:
        #     return random.choice(kan_actions)

        # # discard an effective tile randomly
        # legal_discards = [
        #     a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        # ]
        # effective_discard_types = observation.curr_hand().effective_discard_types()
        # effective_discards = [
        #     a for a in legal_discards if a.tile().type() in effective_discard_types
        # ]
        # if len(effective_discards) > 0:
        #     return random.choice(effective_discards)

        # # if no effective tile exists, discard randomly
        # return random.choice(legal_discards)

        legal_actions = obs.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 予測
        feature = obs.to_features(feature_name="mjx-small-v0")
        # feature = obs.to_features(feature_name="mjx-large-v0")
        self.model.eval()
        with torch.no_grad():
            action_logit = self.model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()
        # print(action_logit)

        # アクション決定
        mask = obs.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)

        # ランダムに取れる行動をする
        return random.choice(obs.legal_actions())


def save_log(obs_dict, env, logs):
    logdir = "logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    now = datetime.now().strftime('%Y%m%d%H%M%S%f')

    os.mkdir(os.path.join(logdir, now))
    for player_id, obs in obs_dict.items():
        with open(os.path.join(logdir, now, f"{player_id}.json"), "w") as f:
            json.dump(json.loads(obs.to_json()), f)
        with open(os.path.join(logdir, now, f"tenho.log"), "w") as f:
            f.write(logs.get_url())
    env.state().save_svg(os.path.join(logdir, now, "finish.svg"))
    with open(os.path.join(logdir, now, f"env.json"), "w") as f:
        f.write(env.state().to_json())

    dir_path = os.path.join(logdir, now)
    return dir_path


if __name__ == "__main__":
    """引数
    -n, --number (int): 何回対局するか
    -l --log (flag): このオプションをつけると対局結果を保存する
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=32,
                        help="number of game iteration")
    parser.add_argument("-l", "--log", action="store_true",
                        help="whether log will be stored")
    args = parser.parse_args()

    logging = args.log
    n_games = args.number

    player_names_to_idx = {
        "player_0": 0,
        "player_1": 1,
        "player_2": 2,
        "player_3": 3,
    }

    agents = [
        MyAgent(),                  # 自作Agent
        mjx.agents.ShantenAgent(),  # mjxに実装されているAgent
        mjx.agents.ShantenAgent(),  # mjxに実装されているAgent
        mjx.agents.ShantenAgent(),  # mjxに実装されているAgent
    ]

    # 卓の初期化
    # env_ = mjx.MjxEnv()
    # obs_dict = env_.reset()
    dir_paths = []
    logs = convert_log.ConvertLog()
    for _ in range(n_games):
        env_ = mjx.MjxEnv()
        obs_dict = env_.reset()
        while not env_.done():
            actions = {}
            for player_id, obs in obs_dict.items():
                actions[player_id] = agents[player_names_to_idx[player_id]].act(
                    obs)
            obs_dict = env_.step(actions)
            if len(obs_dict.keys()) == 4:
                logs.add_log(obs_dict)
        returns = env_.rewards()
        if logging:
            dir_path = save_log(obs_dict, env_, logs)
            dir_paths.append(dir_path)
    print("game has ended")
    if logging:
        custom_evaluate.evaluate_from_arg(dir_paths)

    # from mjx import Action, Observation, State
    # data_path = "json"
    # with open(data_path) as f:
    #     lines = f.readlines()

    # for line in lines:
    #     state = State(line)

    #     for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
    #         obs = Observation._from_cpp_obj(cpp_obs)
    #         feature = obs.to_features(feature_name="mjx-small-v0")

    #         action = Action._from_cpp_obj(cpp_act)
    #         action_idx = action.to_idx()
