import torch
import torch.nn as nn
import torch.nn.functional as F


# model
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size=2048, droput=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # two linear transformation with Relu

    #     self.input_size = input_size
    #     self.output_size = output_size
    #     self.hidden_size = hidden_size

    # def forward(self, x):
    #     out = nn.Linear(x, self.hidden_size)
    #     out = nn.Linear(out, self.output_size)
    #     out = F.relu(out)
    #     return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def attention(self, q, k, v, mask=None, dropout=None):
        "Compute scaled dot product attention"
        d_k = torch.sqrt(k.shape[0])
        scores = torch.matmul(q, k.transpose(0, 1)) / d_k
        scores = F.softmax(scores, dim=1)
        results = torch.matmul(scores, v)
        return results
