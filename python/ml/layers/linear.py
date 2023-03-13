import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # self.bn = nn.BatchNorm1d(output_dim)
        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "leaky_relu":
            self.activation = torch.nn.functional.leaky_relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "x":
            self.activation = lambda x: x  # f(x) = x
        else:
            raise ValueError(f"{activation} is not supported")

    def forward(self, x):
        x = self.linear(x)
        # x = self.bn(x)
        x = self.activation(x)
        return x
