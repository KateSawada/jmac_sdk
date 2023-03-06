"""対局を指定した回数行うスクリプト．対局結果の出力も可能．
"""

import argparse
import os
from datetime import datetime
import json
import random

import mjx
import mjx.agents

from mjx.action import Action
from mjx.const import ActionType
from mjx.env import MjxEnv
from mjx.observation import Observation
from mjx.visualizer.selector import Selector
from mjx.const import EventType, PlayerIdx, TileType
from mjx.event import Event

from server import convert_log
from client.agent import CustomAgentBase


# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．


class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.remining_tiles = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[1,1,1]]
        self.action_mode = "menzen"

    def act(self, obs: mjx.Observation) -> mjx.Action:
        hand = obs.MjxLargeV0().current_hand(obs)
        target = obs.MjxLargeV0().target_tile(obs)
        riichi = obs.MjxLargeV0().under_riichis(obs)
        discarded_tiles = obs.MjxLargeV0().discarded_tiles(obs)
        discarded_from_fand = obs.MjxLargeV0().discarded_from_hand(obs)
        ignored_tiles = obs.MjxLargeV0().ignored_tiles(obs)
        doras = obs.doras()
        

        if not (1 in discarded_tiles[0] or 1 in discarded_tiles[3] or 1 in discarded_tiles[6] or 1 in discarded_tiles[9]):
            self.remining_tiles = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[1,1,1]]
        

        self.remining_tiles = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[1,1,1]]
        for dora in doras:
            opened_dora = 0
            if not dora in [0,9,18,27,31]:
                opened_dora = dora-1
            elif dora==9:
                opened_dora = 17
            elif dora==18:
                opened_dora = 26
            elif dora==27:
                opened_dora = 31
            elif dora==31:
                opened_dora = 33   
            self.remining_tiles[0][opened_dora] -= 1

        check_red_dora = [0,0,0]
        if hand[4][0]==1:
            check_red_dora[0]=1
        elif hand[5][0]==1:
            check_red_dora[1]=1
        elif hand[6][0]==1:
            check_red_dora[2]=1
        self.remining_tiles = [self.remining_tiles[0], [x-y for (x,y) in zip(self.remining_tiles[1],check_red_dora)]]
        for i in range(4):
            self.remining_tiles = [[x-y for (x,y) in zip(self.remining_tiles[0],hand[i])],self.remining_tiles[1]]
        for e in obs.events():
                if e.type() == EventType.DISCARD or e.type() == EventType.TSUMOGIRI:
                    self.remining_tiles[0][e.tile().type()] -= 1
                    if e.tile().is_red() and e.tile().type() == 4:
                        self.remining_tiles[1][0]=0
                    elif e.tile().is_red() and e.tile().type() == 13:
                        self.remining_tiles[1][1]=0
                    elif e.tile().is_red() and e.tile().type() == 22:
                        self.remining_tiles[1][2]=0

                elif e.type() in [
                    EventType.CHI,
                    EventType.PON,
                    EventType.OPEN_KAN,
                ]:
                    for t in e.open().tiles():
                        self.remining_tiles[0][t.type()] -= 1
                        if t.is_red() and t.type() == 4:
                            self.remining_tiles[1][0]=0
                        elif t.is_red() and t.type() == 13:
                            self.remining_tiles[1][1]=0
                        elif t.is_red() and t.type() == 22:
                            self.remining_tiles[1][2]=0
                    self.remining_tiles[0][e.open().last_tile().type()] += 1
                
                elif e.type() in [
                    EventType.CLOSED_KAN,
                ]:
                    for t in e.open().tiles():
                        self.remining_tiles[0][t.type()] -= 1
                        if t.is_red() and t.type() == 4:
                            self.remining_tiles[1][0]=0
                        elif t.is_red() and t.type() == 13:
                            self.remining_tiles[1][1]=0
                        elif t.is_red() and t.type() == 22:
                            self.remining_tiles[1][2]=0

                
                elif e.type() == EventType.ADDED_KAN:
                    t = e.open().last_tile()
                    self.remining_tiles[0][t.type()] -= 1
                    if t.is_red() and t.type() == 4:
                        self.remining_tiles[1][0]=0
                    elif t.is_red() and t.type() == 13:
                        self.remining_tiles[1][1]=0
                    elif t.is_red() and t.type() == 22:
                        self.remining_tiles[1][2]=0
        

        legal_actions = obs.legal_actions()
        kyusyu_actions = [a for a in legal_actions if a.type() == ActionType.ABORTIVE_DRAW_NINE_TERMINALS]
        if len(kyusyu_actions) >= 1:
            assert len(kyusyu_actions) == 1
            self.action_mode = "kyusyu"

        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.action_mode == "kyusyu":
            legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
            ]
            for a in legal_discards:
                print(a.tile().type())


        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        steal_actions = [
            a for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            return pass_action
        """
        if len(steal_actions) >= 1:
            return random.choice(steal_actions)
        """

        added_kan_actions = [
            a for a in legal_actions if a.type() in [ActionType.ADDED_KAN] 
        ]
        if len(added_kan_actions) >= 1:
            assert len(added_kan_actions) == 1
            return added_kan_actions[0]

        closed_kan_actions = [
            a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN]
        ]
        if len(closed_kan_actions) >= 1:
            assert len(closed_kan_actions) == 1
            return closed_kan_actions[0]

        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        ]
        effective_discard_types = obs.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]

        if len(effective_discards) > 0:
            return random.choice(effective_discards)

        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)

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

    player_names_to_idx ={
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
    env_ = mjx.MjxEnv()
    obs_dict = env_.reset()

    my_id = "player_0"
    result_rank = ""

    logs = convert_log.ConvertLog()
    for _ in range(n_games):
        env_ = mjx.MjxEnv()
        obs_dict = env_.reset()
        first_counter = 0
        while not env_.done():
            actions = {}
            for player_id, obs in obs_dict.items():
                actions[player_id] = agents[player_names_to_idx[player_id]].act(obs)
            obs_dict = env_.step(actions)
            if len(obs_dict.keys())==4:
                logs.add_log(obs_dict)
        returns = env_.rewards()
        if env_.rewards()[my_id]==90:
            result_rank = result_rank+"1"
        elif env_.rewards()[my_id]==45:
            result_rank = result_rank+"2"
        elif env_.rewards()[my_id]==0:
            result_rank = result_rank+"3"
        elif env_.rewards()[my_id]==-135:
            result_rank = result_rank+"4"
        else:
            result_rank = result_rank+"?"

        if logging:
            save_log(obs_dict, env_, logs)

    print("rank: "+result_rank)
    print("game has ended")

