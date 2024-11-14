import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from layers.tools.audios import RevSTFT
from layers.tools.utils import *
from layers.blocks.attention import TransformerBlock
from layers.cores.unet import Encoder,Decoder,ResBlock
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn





class net(nn.Module):
   
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.sequence_length = config['seq_len']                                                # Raw sequence length
        self.seq_len,self.embed_dim = 250,512 # for 240,000 length(10sec) audio
        self.num_blocks = config['num_blocks']                                                  # Number of Transformer blocks
        activation_fn = get_activation_fn(config['activation_fn'],in_chn=self.embed_dim)
        norm_fn = get_norm_fn(config['norm_fn'])
        p = config['dropout']
          
        
        # Mapping Net
        self.time_emb = FourierFeatures(1, 64//4//2)
        self.map_layers = nn.Sequential(
            Linear(64//4//2, 64//4//2), # head_dim
            activation_fn,
            Linear(64//4//2,64//4//2),
            activation_fn,
            Linear(64//4//2,64//4//2)
        )

        #self.encoder = BiMambaBlock(dim=self.embed_dim,norm_fn=norm_fn,activation_fn=activation_fn,p=p)
        #self.decoder = BiMambaBlock(dim=self.embed_dim,norm_fn=norm_fn,activation_fn=activation_fn,p=p)
        self.encoder = Encoder(channels=[self.embed_dim,128,64],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.decoder = Decoder(channels=[64,128,self.embed_dim],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embed_dim=64, depth=i + 1, num_heads=4,activation_fn=activation_fn,norm_fn=norm_fn) for i in range(self.num_blocks)]
        )
    
  
        self.last = nn.Sequential(
            ResBlock(channels=self.embed_dim,norm_fn=norm_fn,activation_fn=activation_fn,dropout=0)
        )

    def forward(self,x,sigmas): 
        '''
            x: [batch,seq],
            sigmas: [batch]
        '''
        # Condition Mapping (Timestamp)
        sigmas = self.time_emb(sigmas.unsqueeze(-1))
        sigmas = self.map_layers(sigmas)
        # Condition Mapping
        
        x = x.reshape(x.size(0), self.seq_len, self.embed_dim)        

        x = self.encoder(x)
        
        for trans in self.transformer:
            x = trans(x,sigmas)

        x = self.decoder(x)
  
        x = x.transpose(-1,-2)
        x = self.last(x)
        x = x.transpose(-1,-2)


        return x.reshape(x.size(0),-1)
    



