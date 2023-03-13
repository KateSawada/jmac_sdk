import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalAcc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, actual, predict):
        actual = torch.where(actual > 0.5, 1, 0)
        predict = torch.where(predict > 0.5, 1, 0)
        return torch.sum(torch.prod(actual * predict, axis=1)) / len(actual)

class BinaryAcc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, actual, predict):
        actual = torch.where(actual > 0.5, 1, 0)
        predict = torch.where(predict > 0.5, 1, 0)
        accurate = torch.min(actual + predict, torch.ones_like(actual))
        return torch.sum(accurate) / accurate.numel()
