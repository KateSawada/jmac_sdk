from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.layers.linear import Linear

# A logger for this file
logger = getLogger(__name__)


class YakuPredictor(nn.Module):
    def __init__(self, dense_dims):
        super().__init__()
        self.dense_dims = dense_dims
        self.build_model(dense_dims)

    def build_model(self, dense_dims):
        self.dense_layers = nn.ModuleList()
        for i in range(len(dense_dims) - 2):
            self.dense_layers += [
                Linear(dense_dims[i], dense_dims[i + 1], "relu")
            ]

        # final layer uses sigmoid as a activation function
        self.dense_layers += [
            Linear(dense_dims[-2], dense_dims[-1], "sigmoid")
        ]

    def forward(self, x):
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
        return x
