import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.util_net import RMSNorm,PositionwiseFeedForward,TorchSTFT
from layers.attentionBlock import TransformerBlock
#from nnAudio.features import MelSpectrogram,STFT,iSTFT# instead, use TorchSTFT class.
#from nnAudio.Spectrogram import Griffin_Lim

# config
SR = 8000
N_FFT = 128
WIN_LEN = 128
HOP_LENGTH = 64
CHUNK_LENGTH = 30
N_MELS = 64
MIN=1e-7
MAX=2e+5

        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,p=0.2):
        
        super().__init__()
        self.layer = nn.Sequential(

            nn.Conv1d(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=5, stride=1,
                            padding=2, bias=True),
            RMSNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, 
                            out_channels=out_channels,
                            kernel_size=3, stride=1,
                            padding=1, bias=False),
            
        )
        self.norm = RMSNorm(out_channels,bias=True)
        self.p = p

    def forward(self, x):
        """
            Input shape: [b,L,chn]

            Out shape : [b,L//pool_size,chn]
        """
        x = x.tranpose(-1,-2)
        x = self.norm(self.layer(x) + x)
        x = F.avg_pool1d(x,kernel_size=2,stride=2)
        x = F.dropout(x,p=0.2)
        x = x.transpose(-1,-2)
        return x


class CNNEncoder(nn.Module):
    """
    turn freq_bins into latent vectors.
    """
    def __init__(self,channels:list,):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(in_channels=channels[i],out_channels=channels[i + 1],p=0.2))

        
    def forward(self,x):
        for block in self.blocks:
            x = block(x)

        return x
    

class ConvBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            RMSNorm(out_channels),
            nn.SiLU(),
            nn.ConvTranspose1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=5, stride=1,
                               padding=2, bias=True),
        )
        self.norm = RMSNorm(out_channels, bias=True)
        self.p = p

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.norm(self.layer(x) + x)
        x = F.dropout(x, p=self.p)
        x = x.transpose(-1, -2)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, channels: list):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for i in range(len(channels) - 1, 0, -1):
            self.blocks.append(ConvBlockDecoder(in_channels=channels[i], out_channels=channels[i - 1], p=0.2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x



class net(nn.Module):
   
    def __init__(self,config):
        super().__init__()
        sequence_length = config['seq_len'] 
        num_blocks = config['num_blocks']
        self.sequence_length = sequence_length  
        self.stft = TorchSTFT()
        self.seq_len,self.embed_dim = self.calculate_spectrogram_shape(sequence_length)

        self.encoder = CNNEncoder(channels=[self.embed_dim,64,128,256,512])
        self.decoder = CNNDecoder(channels=[512,256,128,64,self.embed_dim])
        self.transformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.transformer.append(TransformerBlock(embed_dim=512,depth=i+1,num_heads=8))

        self.ff = PositionwiseFeedForward(dims=self.embed_dim)

    def calculate_spectrogram_shape(self,sequence_length):
        # Calculate the number of frames (time dimension)
        with torch.no_grad():
            out,out2 =self.stft.transform(torch.ones((1,sequence_length)))
            ff = torch.cat((out,out2),dim=1) # batch,chn,frames[t]
            ff = ff.transpose(-1,-2) # batch,frames[t], chn
            return ff.shape[1],ff.shape[2]

    def forward(self,x): 
        '''
            input x: [batch,seq]
        '''
        with torch.no_grad():
            mag,angle = self.stft.transform(x) 
            x = torch.cat((mag,angle),dim=1) 
            x = x.transpose(-1,-2) # batch, frames[t], 2 * freq_bins
            


        x = self.encoder(x)
        for trans in self.transformer:
            x = trans(x)

       
        x = self.decoder(x)
        x = self.ff(x)


        spec = x[:,:,:N_FFT // 2 + 1]
        phase = x[:, :, N_FFT // 2 + 1:]
        return self.stft.inverse(spec,phase) 
    



