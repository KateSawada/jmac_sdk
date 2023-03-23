import random
from typing import Dict, List, Optional
import time

import gym
import json
import torch
from torch import Tensor
import mjx
from mjx.agents import RandomAgent, ShantenAgent
from agent.menzenAgent import MenzenAgent

import torch
import torch.optim as optim

from model import MLP, CNN_MLP2

from rf_learning.gymEnv import GymEnv, REINFORCE


# 半荘のスコアと順位に応じたポイントを計算
def calc_rank(rank):
    if rank == -135:
        return 3
    elif rank == 0:
        return 2
    elif rank == 45:
        return 1
    else:
        return 0


if __name__ == "__main__":
    model = CNN_MLP2()
    # model.load_state_dict(torch.load('./model_tenho_100.pth'))
    model.load_state_dict(torch.load("./params/CNN_MLP2/model_3.pth"))

    class MyAgent(mjx.Agent):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def act(self, obs: mjx.Observation) -> mjx.Action:
            """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

            Args:
                obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

            Returns:
                mjx.Action: 実際に取る行動
            """
            legal_actions = obs.legal_actions()
            if len(legal_actions) == 1:
                return legal_actions[0]
            
            # 予測
            feature = Tensor(obs.to_features(feature_name="mjx-small-v0").ravel())
            with torch.no_grad():
                action_logit = self.model.predict(Tensor(feature.ravel()))
            action_proba = torch.sigmoid(action_logit).numpy()

            # アクション決定
            mask = obs.action_mask()
            action_idx = (mask * action_proba).argmax()
            return mjx.Action.select_from(action_idx, legal_actions)


    opt = optim.Adam(model.parameters(), lr=1e-3)
    agent = REINFORCE(model, opt)

    model1 = CNN_MLP2()
    model1.load_state_dict(torch.load("./params/CNN_MLP2/model_3.pth"))
    agent1 = MyAgent(model1)

    model2 = CNN_MLP2()
    model2.load_state_dict(torch.load("./params/CNN_MLP2/model_3.pth"))
    agent2 = MyAgent(model2)

    model3 = CNN_MLP2()
    model3.load_state_dict(torch.load("./params/CNN_MLP2/model_3.pth"))
    agent3 = MyAgent(model3)


    env = GymEnv(
        opponent_agents=[agent1, agent2, agent3],
        reward_type="game_tenhou_7dan",
        done_type="game",
        feature_type="mjx-small-v0",
    )

    # 学習前
    # 報酬（=順位）をカウントする
    # rank_counter = [0 for _ in range(4)]

    # for i in range(100):
    #     obs, info = env.reset()
    #     done = False
    #     while not done:
    #         a = agent.act(obs, info["action_mask"])
    #         # a = take_random_action(info["action_mask"])
    #         obs, r, rank, done, info = env.step(a)

    #     # 順位をカウント
    #     rank_counter[calc_rank(rank)] += 1

    # # 検証
    # print('\n==================================================================\n')
    # print('i: {},  time: {}, R: {}, rank: {}.'.format(0, 0.0, 0, rank_counter))

    # 強化学習～
    for i in range(100):
        start_time = time.time()
        for j in range(100):
            obs, info = env.reset()
            done = False
            R = 0
            while not done:
                a = agent.act(obs, info["action_mask"])
                obs, r, rank, done, info = env.step(a)
                R += r
            #print(R)
            del obs, info
        agent.update_gradient(R)

        # 検証
        print('\n==================================================================\n')
        # 報酬（=順位）をカウントする
        rank_counter = [0 for _ in range(4)]

        for j in range(100):
            obs, info = env.reset()
            done = False
            while not done:
                a = agent.act(obs, info["action_mask"])
                # a = take_random_action(info["action_mask"])
                obs, r, rank, done, info = env.step(a)

            # 順位をカウント
            rank_counter[calc_rank(rank)] += 1
            del obs, info

        #print(i, rank_counter)

        print('i: {},  time: {}, R: {}, rank: {}.'.format(i+1, time.time()-start_time, R, rank_counter))

    # 保存
    torch.save(agent.model.state_dict(), "./params/CNN_MLP2/model_rf_0.pth")