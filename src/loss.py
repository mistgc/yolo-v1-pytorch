import torch
from torch import nn

class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
