from mamba_ssm import Mamba2
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            d_state=64,
            d_conv=4,
            expand=2,
    ):
        super().__init__()
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self,x):
        b,seq,dim = x.shape
        return self.mamba(x)
    


class BackwardMambaBlock(MambaBlock):

    def forward(self,x):
        return self.mamba(x.flip(dims=[1])).flip(dims=[1])
    

class BiMambaBlock(nn.Module):
    # Same logic as BiLSTM
    def __init__(
            self,
            dim,
            d_state=64,
            d_conv=4,
            expand=2,
            norm_fn=nn.LayerNorm,
            activation_fn=nn.SiLU(),
            p=0.1
    ):
        super().__init__()
        self.mambaForward = MambaBlock(dim,d_state,d_conv,expand)
        self.mambaBackward = BackwardMambaBlock(dim,d_state,d_conv,expand)
        self.norm_fn = norm_fn(dim)
        self.reduce = nn.Sequential(
            nn.Linear(dim*2,dim*2),
            nn.Dropout(p=p),
            activation_fn,
            nn.Linear(dim*2,dim)
        )
    def forward(self,x):
        b,seq,dim = x.shape
        x = torch.cat((self.norm_fn(self.mambaBackward(x)) , self.norm_fn(self.mambaBackward(x))),dim=-1)
        return self.reduce(x)