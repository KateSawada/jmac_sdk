import json
import sys
import random
import torch
from torch import nn, Tensor, optim
import mjx
from mjx import Agent, Observation, Action
from mjx.agents import ShantenAgent
from gateway import MjxGateway, to_mjai_tile


class MLP(nn.Module):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=544):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 8,kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(8)

        self.conv4 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(8)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(544*8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

        self.loss_module = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self,x):
        x = x.float()
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

    def predict(self, x):
        x = torch.unsqueeze(x, dim=0)
        x = self.forward(x)
        x = self.softmax(x)
        return x
    


class MLPAgent(Agent):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def act(self, obs: Observation) -> Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        try:
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
        
        except:
            return random.choice(legal_actions)


def main():
    
    model = MLP()
    model.load_state_dict(torch.load('./model.pth'))
    agent = MLPAgent(model)

    player_id = int(sys.argv[1])
    assert player_id in range(4)
    bot = MjxGateway(player_id, agent)

    while True:
        line = sys.stdin.readline().strip()
        resp = bot.react(line)
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
