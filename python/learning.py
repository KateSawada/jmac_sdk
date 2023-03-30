from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch
import mjx
import mjx.agents
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from client.agent import CustomAgentBase


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

inps = np.load("./obs.npy")
tgts = np.load("./actions.npy")
inps = inps.reshape(661069, 16*34)

dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loader = DataLoader(dataset, batch_size=2)

model = MLP()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=model, train_dataloaders=loader)
torch.save(model.state_dict(), './model_0.pth')

class MyAgent(CustomAgentBase):

    def __init__(self) -> None:
        super().__init__()

    def act(self, obs: mjx.Observation) -> mjx.Action:
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

