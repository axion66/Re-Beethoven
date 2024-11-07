import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from layers.tools.audios import RevSTFT
from layers.tools.utils import *
from layers.attn import TransformerBlock
from layers.cnn import Encoder,Decoder
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn
#from nnAudio.features import STFT,iSTFT

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
        self.stft = RevSTFT(config)
        self.istft = iSTFT(n_fft=config['n_fft'],win_length=config['win_len'],hop_length=config['hop_length'],trainable_kernels=True,trainable_window=True)
        self.sequence_length = config['seq_len']                                                # Raw sequence length
        self.seq_len,self.embed_dim = self.calculate_spectrogram_shape(self.sequence_length)    # batch, seq_len, n_fft(embed_dim)
        self.num_blocks = config['num_blocks']                                                  # Number of Transformer blocks
        activation_fn = get_activation_fn(config['activation_fn'],in_chn=self.embed_dim)
        norm_fn = get_norm_fn(config['norm_fn'])
        p = config['dropout']
          
        
        # Mapping Net
        self.time_emb = FourierFeatures(1, 512//8//2)
        self.map_layers = nn.Sequential(
            Linear(512//8//2, 512//8//2), # head_dim
            activation_fn,
            Linear(512//8//2,512//8//2),
            activation_fn,
            Linear(512//8//2,512//8//2)
        )

        self.encoder = Encoder(channels=[self.embed_dim,512,512,512,512],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.decoder = Decoder(channels=[512,512,512,512,self.embed_dim],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embed_dim=512, depth=i + 1, num_heads=8,activation_fn=activation_fn,norm_fn=norm_fn) for i in range(self.num_blocks)]
        )
        self.act = activation_fn
        self.out = PositionwiseFeedForward(dims=self.embed_dim,activation=activation_fn,dropout=p)


        self.prac_lstm = nn.LSTM(self.embed_dim,self.embed_dim)
        



    def calculate_spectrogram_shape(self,sequence_length):
        # Calculate the number of frames (time dimension)
        with torch.no_grad():
            out,out2 =self.stft.transform(torch.ones((1,sequence_length)))
            ff = torch.cat((out,out2),dim=1) # batch,chn,frames[t]
            ff = ff.transpose(-1,-2) # batch,frames[t], chn
            return ff.shape[1],ff.shape[2]

    def forward(self,x,sigmas): 
        '''
            x: [batch,seq],
            sigmas: [batch]
        '''
        # Condition Mapping (Timestamp)
        sigmas = self.time_emb(sigmas.unsqueeze(-1))
        sigmas = self.map_layers(sigmas)
        # Condition Mapping



        with torch.no_grad(): # No need to spread gradient here this is the beginning of x
            mag,angle = self.stft.transform(x) 
            x = torch.cat((mag,angle),dim=1) 
            x = x.transpose(-1,-2) # batch, frames[t], 2 * freq_bins
            
        '''
        x = self.encoder(x)
        
        # for trans in self.transformer:
        #    x = trans(x,sigmas)

        x = self.decoder(x)
        x = self.act(x)
        x = self.out(x)
        '''
        x,_ = self.prac_lstm(x)
        spec = x[:,:, :self.config['n_fft'] // 2 + 1]
        phase = x[:,:, self.config['n_fft'] // 2 + 1:]
        return self.stft.inverse(spec,phase) 
    


