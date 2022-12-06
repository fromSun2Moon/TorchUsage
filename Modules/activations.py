import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GELU(nn.Module):
    def __init__(self, constant=0.044715, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def forward(self, x):
        output = x + self.constant * torch.pow(x, 3)
        output = math.sqrt(2 / math.pi) * output
        output = 1. + torch.tanh(output)
        output = 0.5 * x * output
        return output


if __name__ == "__main__":
    test_in = torch.rand(2,4)
    gelu = GELU(); gelu2 = nn.GELU()
    print(gelu(test_in))
    print(gelu2(test_in))