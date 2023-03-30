import mjx
from mjx import Observation, State, Action 
import torch
from torch import optim, nn, Tensor
import pytorch_lightning as pl
import numpy as np
from client.client import SocketIOClient
from client.agent import CustomAgentBase

class MLP(pl.LightningModule):
    def __init__(self, obs_size=548, n_actions=181, hidden_size=128):
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
model.load_state_dict(torch.load('model_mono.pth'))
class MonoAgent(CustomAgentBase):

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        feature = observation.to_features(feature_name="mjx-small-v0")
        riich=observation.MjxLargeV0.under_riichis(observation)
        r=np.array([0,0,0,0])
        for i in range(4):
            if riich[i][0]==1:
                r[i]=1
            else:
                r[i]=0
        feature=np.append(feature,r)
        with torch.no_grad():
            action_logit = model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()
        
        # アクション決定
        mask = observation.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)


if __name__ == "__main__":
    # 4人で対局する場合は，4つのSocketIOClientで同一のサーバーに接続する．
    my_agent = MonoAgent()  # 参加者が実装したプレイヤーをインスタンス化

    sio_client = SocketIOClient(
        ip='localhost',
        port=5000,
        namespace='/test',
        query='secret',
        agent=my_agent,  # プレイヤーの指定
        room_id=123,  # 部屋のID．4人で対局させる時は，同じIDを指定する．
        )
    # SocketIO Client インスタンスを実行
    sio_client.run()
    sio_client.enter_room()



