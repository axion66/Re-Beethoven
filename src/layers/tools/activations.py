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


class SnakeBeta(nn.Module):
    # from ttps://github.com/NVIDIA/BigVGAN/blob/main/activations.py

    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta = x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


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
            GELU,
            Snake(For sequential data.)
        
        Default: SiLU
    """
    if act == "SwiGLU":
        assert isinstance(in_chn,int)
        return SwiGLU(in_chn=in_chn)
    elif act == "xATLU":
        return xATLU()
    elif act == "ReLU":
        return nn.ReLU()
    elif act == "SiLU" or act == "Swish" or act == 'silu':
        return SiLU()
    elif act == "GELU":
        return nn.GELU(approximate='tanh')
    elif act == "snake":
        assert isinstance(in_chn,int)
        return SnakeBeta(in_chn)
    else:
        return SiLU()