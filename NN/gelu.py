import math

import torch
import torch.nn as nn


class GELU(nn.module):
    def __init__(self, constant=0.044715, *args, **kwargs):
        super(self, GELU).__init__(*args, **kwargs)
        self.x = x
        self.constant = constant

    def forward(self, x):
        output = x + self.constant * torch.pow(x, 3)
        output = torch.sqrt(2 / math.pi) * output
        output = 1. + torch.tanh(output)
        output = 0.5 * x * output
        return output
