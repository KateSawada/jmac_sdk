import random 
import mjx
import torch
from torch import Tensor

import argparse
import os
from datetime import datetime
import json
import random
import mjx.agents

from server import convert_log
from client.agent import CustomAgentBase

from workspace.model import MLP

# モデルの読み込み
# state_dict()はパラメータのみを保存するため、モデル構造を定義してから読み込む
model = MLP()
model.load_state_dict(torch.load('workspace/model_shanten_100.pth'))

# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．
class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        legal_actions = obs.legal_actions()
        if len(legal_actions == 1):
            return legal_actions[0]
        
        # 予測
        feature = obs.to_features(feature_name="mjx-small-v0")
        with torch.no_grad():
            action_logit = model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()

        # アクション決定
        mask = obs.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)


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

def calc_score(tens):
    tmp = {}
    for i in range(4):
        tmp['player_'+str(i)] = tens[i]
    sorted_tens = sorted(tmp.items(), key=lambda x:x[1], reverse=True)

    scores = {}
    scores[sorted_tens[0][0]] = round((sorted_tens[0][1]-30000)/1000 + 50, 1)
    scores[sorted_tens[1][0]] = round((sorted_tens[1][1]-30000)/1000 + 10, 1)
    scores[sorted_tens[2][0]] = round((sorted_tens[2][1]-30000)/1000 - 10, 1)
    scores[sorted_tens[3][0]] = round((sorted_tens[3][1]-30000)/1000 - 30, 1)

    return scores



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

    scores = {
        "player_0": 0.0,
        "player_1": 0.0,
        "player_2": 0.0,
        "player_3": 0.0,
    }


    # 卓の初期化
    env_ = mjx.MjxEnv()
    obs_dict = env_.reset()

    logs = convert_log.ConvertLog()
    for _ in range(n_games):
        # 卓の初期化（ここでやらないと毎回同じ結果になってる）
        obs_dict = env_.reset()
        while not env_.done():
            actions = {}
            for player_id, obs in obs_dict.items():
                actions[player_id] = agents[player_names_to_idx[player_id]].act(obs)
            obs_dict = env_.step(actions)
            if len(obs_dict.keys())==4:
                logs.add_log(obs_dict)
        returns = env_.rewards()

        # 各半荘の結果を表示
        obs_json = json.loads(obs_dict['player_0'].to_json())
        cur_tens = obs_json["roundTerminal"]['finalScore']['tens']
        cur_scores = calc_score(cur_tens)

        for player_id, score in cur_scores.items():
            # print(player_id, score)
            scores[player_id] += score
        # print('======================================================')
        if logging:
            save_log(obs_dict, env_, logs)
        
    print("game has ended")

    for player_id, score in scores.items():
        print(player_id, round(score, 1))

