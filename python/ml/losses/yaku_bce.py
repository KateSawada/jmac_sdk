import torch
import torch.nn as nn
import torch.nn.functional as F


class YakuBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, y, y_):
        return self.loss(y, y_)
