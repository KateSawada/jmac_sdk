# %%

import random
from typing import Dict, List, Optional

import gym

import mjx
from mjx.agents import RandomAgent

# gym must be 0.25.0+ to use reset(return_info=True)
gym_version = [int(x) for x in gym.__version__.split(".")]
assert (
    gym_version[0] > 0 or gym_version[1] >= 25
), f"Gym version must be 0.25.0+ to use reset(infos=True): {gym.__version__}"

# %%


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
        r = self.mjx_env.rewards(self.reward_type)[self.target_player]
        feat = obs.to_features(self.feature_type)
        mask = obs.action_mask()

        return feat, r, done, {"action_mask": mask}


# %%
def take_random_action(action_mask) -> int:
    legal_idxs = []
    for i in range(len(action_mask)):
        if action_mask[i] > 0.5:
            legal_idxs.append(i)
    return random.choice(legal_idxs)


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


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
        logits = self.model(observation)
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
        loss = -R * self.log_probs  # - self.entropy * 0.01
        loss.backward()
        self.opt.step()
        self.log_probs = 0
        self.entropy = 0

import torch
import mjx
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import mjx.agents

from server import convert_log
from client.agent import CustomAgentBase
# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．import random


class MLP(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=544):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.net(x.float())

model = MLP()
model.load_state_dict(torch.load('./model_0.pth'))

class MyAgent(CustomAgentBase):

    def __init__(self) -> None:
        super().__init__()

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        legal_actions = obs.legal_actions()
        if len(legal_actions) == 1:
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

# %%
agent1 = MyAgent()
agent2 = MyAgent()
agent3 = MyAgent()
env = GymEnv(
    opponent_agents=[agent1, agent2, agent3],
    reward_type="game_tenhou_7dan",
    done_type="game",
    feature_type="mjx-small-v0",
)

opt = optim.Adam(model.parameters(), lr=1e-3)
agent0 = REINFORCE(model, opt)
counter = {
    90 : 0,
    45 : 0,
    0 : 0,
    -135 : 0,
}
for i in range(100):
    obs, info = env.reset()
    done = False
    R = 0
    while not done:
        a = agent0.act(obs, info["action_mask"])
        obs, r, done, info = env.step(a)
        R += r
    counter[r] += 1
    agent0.update_gradient(R)
print(counter)