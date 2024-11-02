import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.tools.audios import RevSTFT
from layers.tools.utils import RMSNorm,PositionwiseFeedForward,Linear
from layers.attentionBlock import TransformerBlock
#from nnAudio.features import MelSpectrogram,STFT,iSTFT# instead, use TorchSTFT class.
#from nnAudio.Spectrogram import Griffin_Lim



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=5, stride=1,
                               padding=2, bias=True)
        self.rms1 = RMSNorm(out_channels)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.rms2 = RMSNorm(out_channels)
        self.p = p

    def forward(self, x):
        # Input shape: [batch, length, channels]
        x = x.transpose(-1, -2)  # Change to [batch, channels, length]
        x = self.conv1(x)
        x = x.transpose(-1, -2)
        x = self.rms1(x)
        x = x.transpose(-1, -2)
        x = self.activation(x)
        x = self.conv2(x)
        x = x.transpose(-1, -2)
        x = self.rms2(x)
        x = x.transpose(-1, -2)

        #x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=self.p)

        x = x.transpose(-1, -2) 
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
        super(ConvBlockDecoder, self).__init__()

        # Define each layer separately
        self.conv_transpose1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.rms1 = RMSNorm(out_channels)
        self.activation = nn.SiLU()
        self.conv_transpose2 = nn.ConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5, stride=1,
            padding=2, bias=True
        )
        self.rms2 = RMSNorm(out_channels, bias=True)
        
        self.p = p

    def forward(self, x):
        # Input shape: [batch, length, channels]
        x = x.transpose(-1, -2)  # Change to [batch, channels, length]

        # Forward pass through each layer
        x = self.conv_transpose1(x)
        x = x.transpose(-1,-2)
        x = self.rms1(x)
        x = x.transpose(-1,-2)
        x = self.activation(x)
        x = self.conv_transpose2(x)
        x = x.transpose(-1,-2)
        x = self.rms2(x)
        x = x.transpose(-1,-2)

        # Apply dropout
        x = F.dropout(x, p=self.p)

        # Transpose back to original shape
        x = x.transpose(-1, -2)  # Change back to [batch, length, channels]
        return x

class CNNDecoder(nn.Module):
    def __init__(self, channels: list):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlockDecoder(in_channels=channels[i], out_channels=channels[i + 1], p=0.2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class FourierFeatures(nn.Module):
    # from NCSN++.
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std) 
        # Don't understand 1 -> embed where transition is done w/ noise that makes model harder to understand.
        # the idea of expanding (1,) shape into sin,cos seems too much. even though it has some smoothing effect.
        #self.proj = nn.Linear(in_features=in_features,out_features=out_features//2,bias=False)

    def forward(self, x):
        f = 2 * 3.141592653589793 * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)





class net(nn.Module):
   
    def __init__(self,config):
        super().__init__()
        self.config = config
        sequence_length = config['seq_len'] 
        num_blocks = config['num_blocks']
        self.sequence_length = sequence_length  
        self.stft = RevSTFT(config)
        self.seq_len,self.embed_dim = self.calculate_spectrogram_shape(sequence_length)

        self.encoder = CNNEncoder(channels=[self.embed_dim,512,512])
        self.decoder = CNNDecoder(channels=[512,512,self.embed_dim])
        self.transformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.transformer.append(TransformerBlock(embed_dim=512,depth=i+1,num_heads=8))

        self.ff = PositionwiseFeedForward(dims=self.embed_dim)


        #mapping
        self.time_emb = FourierFeatures(1, self.embed_dim)
        self.map_layers = nn.Sequential(
            Linear(self.embed_dim, 512, bias=False),
            nn.SiLU(),
            Linear(512,512),
            nn.SiLU(),
            Linear(512,512)
        )


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
        # mapping
        sigmas = self.time_emb(sigmas.unsqueeze(-1))
        sigmas = self.map_layers(sigmas)



        #with torch.no_grad():
        mag,angle = self.stft.transform(x) 
        x = torch.cat((mag,angle),dim=1) 
        x = x.transpose(-1,-2) # batch, frames[t], 2 * freq_bins
        x = self.encoder(x)
        for trans in self.transformer:
            x = trans(x,sigmas)
        x = self.decoder(x)
        x = x[:,:self.seq_len,:]
        x = self.ff(x)

        spec = x[:,:, :self.config['n_fft'] // 2 + 1]
        phase = x[:,:, self.config['n_fft'] // 2 + 1:]
        
        return self.stft.inverse(spec,phase) 
    



