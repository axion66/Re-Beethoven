import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    # Tri-RMSNorm can be used later. 
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


# Traditional Problem w/ Linear is that the weight does not reflect the input. Liquidity? is needed.

class Linear(nn.Module): # x * adaLN(x)
    def __init__(self, in_chn,out_chn,bias=True):
        self.linear = nn.Linear(in_chn,out_chn*3,bias=False)
        self.bias = nn.Parameter(torch.ones(1))
    def forward(self,x):
        x,scale,shift = self.linear(x).chunk(chunks=3,dim=-1)

        return (1 + scale) * x + self.bias * shift

class PositionwiseFeedForward(nn.Sequential):

    def __init__(self, dims: int,activation=nn.SiLU(),rate: int = 4, dropout: float = 0.2):
        super().__init__(
            Linear(dims,dims * rate),
            activation,
            nn.Dropout(dropout),
            Linear(dims * rate, dims))



