import random
from typing import Dict, List, Optional

import gym

import json

import mjx
from mjx.agents import RandomAgent, ShantenAgent
from agent.menzenAgent import MenzenAgent

import torch
import torch.optim as optim

from model import MLP, CNN_MLP

from rf_learning.gymEnv import GymEnv, REINFORCE


# 半荘のスコアと順位に応じたポイントを計算
def calc_score(r):
    if r == -135:
        return -100
    elif r == 0:
        return -20
    elif r == 45:
        return 20
    else:
        return 100

agent1 = ShantenAgent()
agent2 = MenzenAgent()
agent3 = RandomAgent()


env = GymEnv(
    opponent_agents=[agent1, agent2, agent3],
    reward_type="game_tenhou_7dan",
    done_type="game",
    feature_type="mjx-small-v0",
)

model = MLP()
# model.load_state_dict(torch.load('./model_tenho_100.pth'))
model.load_state_dict(torch.load("./params/model_tenho_75000.pth"))


opt = optim.Adam(model.parameters(), lr=1e-3)
agent = REINFORCE(model, opt)

# 強化学習～
avg_R = 0.0
for i in range(1):
    R = 0
    for j in range(1):
        obs, info = env.reset()
        done = False
        while not done:
            a = agent.act(obs, info["action_mask"])
            obs, r, done, info = env.step(a)
            if len(obs.keys())==4:
                print(obs['player_0'].tens())
        
        # 各半荘の結果を表示
        R += calc_score(r)
    
    agent.update_gradient(R)
    print(i,' : ', R, flush=True)

#torch.save(agent.model.state_dict(), "./params/model_tenho_2_rf.pth")


# 報酬（=順位）をカウントする
# rank_counter = {}

# for i in range(100):
#     obs, info = env.reset()
#     done = False
#     while not done:
#         a = agent.act(obs, info["action_mask"])
#         # a = take_random_action(info["action_mask"])
#         obs, r, done, info = env.step(a)

#     # 順位をカウント
#     if r in rank_counter:
#         rank_counter[r] += 1
#     else: 
#         rank_counter[r] = 1

# print(rank_counter)