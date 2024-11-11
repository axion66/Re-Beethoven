import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from layers.tools.audios import RevSTFT
from layers.tools.utils import *
from layers.attn import TransformerBlock
from layers.cores.core_Unet import Encoder,Decoder,ResBlock
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn


class FourierFeatures(nn.Module):
    # from NCSN++.
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std) 

    def forward(self, x):
        f = 2 * 3.141592653589793 * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)




class net(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config['sr'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_len'],
            n_mels=80,
            center=True,
            pad_mode='reflect',
            power=2.0,
            f_min=0.0,
            f_max=8000
        )

        self.sequence_length = config['seq_len']                                                # Raw sequence length
        self.seq_len,self.embed_dim = self.calculate_mel(torch.zeros((1,self.sequence_length)))
        #250,512 # for 240,000 length(10sec) audio
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
        self.encoder = Encoder(channels=[self.embed_dim,64,64],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.decoder = Decoder(channels=[64,64,self.embed_dim],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embed_dim=64, depth=i + 1, num_heads=4,activation_fn=activation_fn,norm_fn=norm_fn) for i in range(self.num_blocks)]
        )
    
  
        self.last = nn.Sequential(
            ResBlock(channels=self.embed_dim,norm_fn=norm_fn,activation_fn=activation_fn,dropout=0)
        )


    def calculate_mel(self,x):
        mel_spec = torch.log(self.mel_transform(x)).transpose(-1,-2)
        return mel_spec.shape[1],mel_spec.shape[2]
    
    def forward(self,x,sigmas): 
        '''
            x: [batch,seq],
            sigmas: [batch]
        '''
        #print(x.shape)
        #print(sigmas.shape)
        # Condition Mapping (Timestamp)
        sigmas = self.time_emb(sigmas.unsqueeze(-1))
        sigmas = self.map_layers(sigmas)
        # Condition Mapping
        
        #x = x.reshape(x.size(0), self.seq_len, self.embed_dim)        
        #x = torch.log(self.mel_transform(x)).transpose(-1,-2) # b,seq,dim <- will be implemented in Denoise function.
        x = self.encoder(x)
        
        for trans in self.transformer:
            x = trans(x,sigmas)

        x = self.decoder(x)
  
        x = x.transpose(-1,-2)
        x = self.last(x)
        x = x.transpose(-1,-2)

        return x
        #return x.reshape(x.size(0),-1)
    



