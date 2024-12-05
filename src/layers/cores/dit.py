import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from layers.tools.utils import FourierFeatures,Linear
from layers.blocks.attention import TransformerBlock, DiTBlock
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn
from layers.autoencoder.vae import AudioAutoencoder,AutoEncoderWrapper
from einops import rearrange
from einops.layers.torch import Rearrange
from layers.tools.activations import *

class DiT(nn.Module):
   
    def __init__(
        self,
        config,
        pretrained_autoencoder: AutoEncoderWrapper = AutoEncoderWrapper(),
    ):
        super().__init__()

        self.autoencoder = pretrained_autoencoder
        self.autoencoder.requires_grad_(False)
        
        _, self.latents_dim, self.seq_len = self.autoencoder.get_latents_shape(example=torch.ones(1, config['seq_len']))
        
        self.embed_dim = config['embed_dim']       
                                                       
        activation_fn = get_activation_fn(config['activation_fn'], in_chn=self.embed_dim)    # hard-coding is better..?
        norm_fn = get_norm_fn(config['norm_fn'])
        
        self.map_sigma = nn.Sequential(
            FourierFeatures(1, self.embed_dim),
            Linear(self.embed_dim, self.embed_dim*4), 
            nn.SiLU(),
            Linear(self.embed_dim*4, self.embed_dim),
            nn.SiLU()
        )

        self.dit = DiTBlock(
            num_blocks = config['num_blocks'],
            latents_dim = self.latents_dim,
            embed_dim = self.embed_dim,
            num_heads = config['num_heads'],
            norm_fn = norm_fn,
            p = config['dropout']
        )
        
    def sigma_transform(self, sigmas):
        while (sigmas.dim() != 2):  # batch, 1
            sigmas = sigmas.unsqueeze(-1) if sigmas.dim() == 1 else sigmas.squeeze(-1)
        return self.map_sigma(sigmas)
        
    def forward(self,x,sigmas): 
        '''
            x: [batch,audio_len],
            sigmas: [batch] or [batch, 1] or [batch, 1, 1]
        '''

        
        l = self.autoencoder.encode(x)

        l = l.transpose(-1, -2)

        l = self.dit(l, self.sigma_transform(sigmas))

        l = l.transpose(-1, -2)

        x = self.autoencoder.decode(l)
        
        return x
    



