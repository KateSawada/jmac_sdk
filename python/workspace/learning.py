import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from model import MLP

inps = np.load("./shanten_obs.npy")
tgts = np.load("./shanten_actions.npy")

# ロガー
tb_logger = pl_loggers.TensorBoardLogger('logs/')

dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loader = DataLoader(dataset, batch_size=2)

model = MLP()
trainer = pl.Trainer(max_epochs=1, logger=tb_logger)
trainer.fit(model, loader)
torch.save(model.state_dict(), "./model_shanten_100.pth")

