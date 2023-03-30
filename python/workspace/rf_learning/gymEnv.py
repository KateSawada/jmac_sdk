import random
from typing import Dict, List, Optional

import gym
import numpy as np
import json

import mjx

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# gym must be 0.25.0+ to use reset(return_info=True)
gym_version = [int(x) for x in gym.__version__.split(".")]
assert (
    gym_version[0] > 0 or gym_version[1] >= 25
), f"Gym version must be 0.25.0+ to use reset(infos=True): {gym.__version__}"

class GymEnv(gym.Env):
    def __init__(
        self, opponent_agents: List[mjx.Agent], reward_type: str, done_type: str, feature_type: str
    ) -> None:
        super().__init__()
        self.opponen_agents = {}
        assert len(opponent_agents) == 3
        for i in range(3):
            self.opponen_agents[f"player_{i+1}"] = opponent_agents[i]
        self.reward_type = reward_type
        self.done_type = done_type
        self.feature_type = feature_type

        self.target_player = "player_0"
        self.mjx_env = mjx.MjxEnv()
        self.curr_obs_dict: Dict[str, mjx.Observation] = self.mjx_env.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = True,
        options: Optional[dict] = None,
    ):
        assert return_info
        if self.mjx_env.done("game"):
            self.curr_obs_dict = self.mjx_env.reset()

        # skip other players' turns
        while self.target_player not in self.curr_obs_dict:
            action_dict = {
                player_id: self.opponen_agents[player_id].act(obs)
                for player_id, obs in self.curr_obs_dict.items()
            }
            self.curr_obs_dict = self.mjx_env.step(action_dict)
            # game ends without player_0's turn
            if self.mjx_env.done("game"):
                self.curr_obs_dict = self.mjx_env.reset()

        assert self.target_player in self.curr_obs_dict
        obs = self.curr_obs_dict[self.target_player]
        feat = obs.to_features(self.feature_type)
        mask = obs.action_mask()
        return feat, {"action_mask": mask}

    def step(self, action: int):
        # prepare action_dict
        action_dict = {}
        legal_actions = self.curr_obs_dict[self.target_player].legal_actions()
        action_dict[self.target_player] = mjx.Action.select_from(action, legal_actions)
        for player_id, obs in self.curr_obs_dict.items():
            if player_id == self.target_player:
                continue
            action_dict[player_id] = self.opponen_agents[player_id].act(obs)

        # update curr_obs_dict
        self.curr_obs_dict = self.mjx_env.step(action_dict)

        # skip other players' turns
        while self.target_player not in self.curr_obs_dict:
            action_dict = {
                player_id: self.opponen_agents[player_id].act(obs)
                for player_id, obs in self.curr_obs_dict.items()
            }
            self.curr_obs_dict = self.mjx_env.step(action_dict)

        # parepare return
        assert self.target_player in self.curr_obs_dict, self.curr_obs_dict.items()
        obs = self.curr_obs_dict[self.target_player]
        done = self.mjx_env.done(self.done_type)
        r = 0
        rank = self.mjx_env.rewards(self.reward_type)[self.target_player]

        # obsから局の情報を取得
        obs_json = json.loads(obs.to_json())
        if 'roundTerminal' in obs_json:
            if "who" in obs_json:
                player_id = obs_json['who']
            else:
                player_id = 0
            if obs_json['publicObservation']['events'][-1]["type"] == "EVENT_TYPE_RON":
                # 点数変化
                tenChanges = np.array(obs_json['roundTerminal']['wins'][0]['tenChanges'])
                # 放銃に対するマイナス
                if tenChanges[player_id] < 0:
                    # 点数状況
                    pre_score = np.array(obs_json['publicObservation']['initScore']['tens'])
                    # print(obs_json['roundTerminal']['wins'][0])
                    # pre_rank = np.argsort(-pre_score)
                    cur_score = pre_score + tenChanges
                    cur_rank = np.argsort(-cur_score)

                    # print(pre_rank[player_id])
                    # print(cur_rank[player_id])
                    for i in range(4):
                        if cur_rank[i] == player_id:
                            # 着順と点数に応じた報酬
                            r = i*(tenChanges[player_id]/1000.0)
                            # print(i)
                            break


                    # print(r)




        feat = obs.to_features(self.feature_type)
        mask = obs.action_mask()

        return feat, r, rank, done, {"action_mask": mask}


# # %%
# def take_random_action(action_mask) -> int:
#     legal_idxs = []
#     for i in range(len(action_mask)):
#         if action_mask[i] > 0.5:
#             legal_idxs.append(i)
#     return random.choice(legal_idxs)



class REINFORCE:
    def __init__(self, model: nn.Module, opt: optim.Optimizer) -> None:
        self.model = model
        self.log_probs = 0
        self.entropy = 0
        self.opt: optim.Optimizer = opt

    def act(self, observation, action_mask):
        observation = torch.from_numpy(observation).flatten().float()
        mask = torch.from_numpy(action_mask)
        # print("mask", mask)
        # probs = self.model(observation)
        # logits = self.model(observation)
        logits = self.model.predict_rf(observation)
        logits -= (1 - mask) * 1e9
        dist = Categorical(logits=logits)
        # dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print("probs", dist.probs)
        self.entropy += dist.entropy()  # (num_envs)
        self.log_probs += log_prob
        assert action_mask[action.item()] == 1, action_mask[action.item()]
        return int(action.item())

    def update_gradient(self, R):
        self.opt.zero_grad()
        loss = R * self.log_probs*1e-6 #- self.entropy * 0.01
        loss.backward()
        self.opt.step()
        self.log_probs = 0
        self.entropy = 0
        del loss