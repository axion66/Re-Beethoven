import torch
import torch.nn as nn
import math
def exists(val):
    return val is not None

class Linear(nn.Module): # x * adaLN(x)
    def __init__(self, in_chn,out_chn,bias=True):
        super().__init__()
        self.linear = nn.Linear(in_chn,out_chn,bias=bias)
    def forward(self,x):

        return self.linear(x)

class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dims: int, activation=nn.SiLU(),rate: int = 4, dropout: float = 0.2):
        super().__init__(
            Linear(dims,dims * rate),
            activation,
            nn.Dropout(dropout),
            Linear(dims * rate, dims)
        )

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


\
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

def transpose_norm_fn(norm):
    return nn.Sequential(
        Transpose(1,-1),
        norm,
        Transpose(1,-1)
    )

