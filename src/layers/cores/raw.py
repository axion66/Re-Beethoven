import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from layers.tools.utils import FourierFeatures,Linear
from layers.blocks.attention import TransformerBlock
from layers.cores.unet import Encoder,Decoder,ResBlock
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn
from layers.autoencoder.vae import AudioAutoencoder
from einops import rearrange, reduce



class net(nn.Module):
   
    def __init__(self,config):
        super().__init__()

        self.autoencoder = AudioAutoencoder(
            io_channels=1,
            in_channels=1,
            out_channels=1,
            sample_rate=config['sr'],
            downsampling_ratio=config['downsampling_ratio'],
        )
        self.embed_dim,self.seq_len = self.calculate_latent_shape(config['seq_len'])
        
        
                                                       
        activation_fn = get_activation_fn(config['activation_fn'],in_chn=self.embed_dim)
        norm_fn = get_norm_fn(config['norm_fn'])
        p = config['dropout']
        num_heads = config['num_heads']
        sigma_dim = self.embed_dim
        
        self.map_sigma = nn.Sequential(
            FourierFeatures(1, sigma_dim),
            Linear(sigma_dim, sigma_dim*4), 
            nn.SiLU(),  # can't use SnakeBeta as dimention of (B,C,T) is required, but in this case we have (B, C)
            Linear(sigma_dim*4, sigma_dim), 
        )

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                              embed_dim=self.embed_dim,
                              depth=i + 1,
                              num_heads=num_heads,
                              sigma_dim=sigma_dim,
                              norm_fn=norm_fn,
                              activation_fn=activation_fn,
                              p=p,
                              ) for i in range(config['num_blocks'])
            ]
        )
    
    def calculate_latent_shape(self,audio_len):
        with torch.no_grad():
            example_x = torch.ones((1,audio_len))
            latents = self.autoencoder.encode(example_x)
            decoded = self.autoencoder.decode(latents)
            assert example_x.shape != decoded.shape, f"input shape{example_x.shape} should match with decoded{decoded.shape} shape."
        return latents.shape[1], latents.shape[2]   # latents shape: {batch, dim, len}
    
    
    def forward(self,x,sigmas): 
        '''
            x: [batch,audio_len],
            sigmas: [batch] or [batch, 1] or [batch, 1, 1]
        '''
        
        
        
        
        
        x = self.autoencoder.encode(x)  # batch, dim, seq
        
        while (sigmas.dim() != 2):  # batch, 1
            sigmas = sigmas.unsqueeze(-1) if sigmas.dim() == 1 else sigmas.squeeze(-1)
        sigmas = self.map_sigma(sigmas)
         
        x = rearrange(x, "b d l -> b l d")
        
        for block in self.transformer:
            x = block(x,sigmas)
            
        x = rearrange(x, "b l d -> b d l")
        
        x = self.autoencoder.decode(x)

        return x
    



