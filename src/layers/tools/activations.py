import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class SwiGLU(nn.Module): 
    # Raffel et al. adding additional last layer will turn this into FFNSwiGLU.
    def __init__(self,in_chn):
        super().__init__()
        self.linear = nn.Linear(in_chn,in_chn*2,bias=False)
    def forward(self, x):
        x1,x2 = torch.chunk(self.linear(x),chunks=2,dim=-1)
        
        return F.silu(x1) * x2


class xATLU(nn.Module):
    def __init__(self):
        super(xATLU, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.half_pi = math.pi / 2
        self.inv_pi = 1 / math.pi

    def forward(self, x):
        gate = (torch.arctan(x) + self.half_pi) * self.inv_pi
        return x * (gate * (1 + 2 * self.alpha) - self.alpha)




class SiLU(nn.Module): # for pytorch <= 1.5
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation_fn(act:Optional[str],in_chn=Optional[int]):
    """
        Options:
            SwiGLU(required param in_chn),
            xATLU,
            ReLU,
            SiLU,
            GELU
        
        Default: SiLU
    """
    if act == "SwiGLU":
        assert isinstance(in_chn,int)
        return SwiGLU(in_chn=in_chn)
    elif act == "xATLU":
        return xATLU()
    elif act == "ReLU":
        return nn.ReLU()
    elif act == "SiLU" or act == "Swish":
        return SiLU()
    elif act == "GELU":
        return nn.GELU(approximate='tanh')
    else:
        return SiLU()