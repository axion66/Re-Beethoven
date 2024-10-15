from layers.minLSTM_block import MinLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from layers.util_net import RMSNorm,decompose,PositionwiseFeedForward,TorchSTFT
#from nnAudio.features import MelSpectrogram,STFT,iSTFT# instead, use TorchSTFT class.
#from nnAudio.Spectrogram import Griffin_Lim

# config
SR = 8000
N_FFT = 512
WIN_LEN = 512
HOP_LENGTH = 256
CHUNK_LENGTH = 30
N_MELS = 64
MIN=1e-7
MAX=2e+5
        
class TransformerBlock(nn.Module):
    '''
        A weird combination of iTransformer, DLinear, DiffTransformer, and MinLSTM.
    '''
    def __init__(self, embed_dim, input_chn, layer_index, device='cuda:0',lstm=False):
        '''
            Mixed diff attention with MinLSTM and trend, seasonal decomposition.
        
            Turn [batch,seq,chn] -> [batch,seq{->embed_dim},512]
        '''
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.25
        self.lstm = lstm

        
        self.ln_attn = RMSNorm(embed_dim).to(self.device) # RMS is center-variant (so that maximize the uses of bias)
        self.decomposition = decompose(kernel_size=5).to(self.device)

        if lstm:
            self.q12 = MinLSTM(input_size=input_chn, hidden_size=input_chn*2).to(self.device)
            self.k12 =  MinLSTM(input_size=input_chn, hidden_size=input_chn*2).to(self.device)
        else: # linear
            self.q12 = nn.Linear(embed_dim,embed_dim*2).to(self.device)
            self.k12 = nn.Linear(embed_dim,embed_dim*2).to(self.device)
        
        self.v = nn.Linear(embed_dim,embed_dim).to(self.device)

            
        self.lambda_init = self.set_lambda(torch.Tensor([layer_index]).to(self.device)) if layer_index is not None else 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0, std=0.1).to(self.device))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0, std=0.1).to(self.device))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0, std=0.1).to(self.device))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0, std=0.1).to(self.device))

        
        self.norm = RMSNorm(embed_dim).to(self.device)
        self.ln_ff = RMSNorm(embed_dim).to(self.device)
        self.ff = PositionwiseFeedForward(dims=embed_dim,).to(self.device)


    def set_lambda(self, layer_index):
        return 0.8 - 0.6 * torch.exp(-0.3 * layer_index)

    def forward(self, x):
        '''
        Get x: [batch,seq,chn]
        '''

        a = self.ln_attn(x.transpose(-1,-2)).transpose(-1,-2) # post-layernorm so that we can use bias in the linaer/lstm net.
        trend,seasonal = self.decomposition(a)
        a = self.attention(trend,seasonal,a)
        x = x + a
        x = x + self.ff(self.ln_ff(x.transpose(-1,-2))).transpose(-1,-2)
        return x
    
    def attention(self, trend,seasonal,a):
        if self.lstm:
            trendQ = self.q12(trend)
            seasonalS = self.k12(seasonal)
            q1, q2 = torch.chunk(trendQ, chunks=2, dim=-1)  # b s d
            q1, q2 = q1.transpose(-1,-2),q2.transpose(-1,-2) # b d s
            k1, k2 = torch.chunk(seasonalS, chunks=2, dim=-1) # b s d
        else: # linaer
            trendQ = self.q12(trend.transpose(-1,-2))
            seasonalS = self.k12(seasonal.transpose(-1,-2))
            q1, q2 = torch.chunk(trendQ, chunks=2, dim=-1)  # b d s
            k1, k2 = torch.chunk(seasonalS, chunks=2, dim=-1) # b d s
            k1, k2 = k1.transpose(-1,-2), k2.transpose(-1,-2) # b s d

        v = self.v(a.transpose(-1, -2)) # b d s

        a1 = torch.bmm(q1, k1) * self.scale # b d d
        a2 = torch.bmm(q2, k2) * self.scale # b d d 
        d = F.softmax(a1, dim=-1) - self.get_lambda() * F.softmax(a2, dim=-1)  # b d d
        
        attn = self.norm(torch.bmm(d, v)) # b d s
        attn *= (1 - self.lambda_init)
        return attn.transpose(-1,-2) # b s d

    def get_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        return lambda_full



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,p=0.2):
        
        super().__init__()
        self.layer = nn.Sequential(

            nn.Conv1d(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=5, stride=1,
                            padding=2, bias=False),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, 
                            out_channels=out_channels,
                            kernel_size=3, stride=1,
                            padding=1, bias=False),
                        nn.SiLU(),
            nn.LayerNorm(out_channels)
        )
        self.ln = nn.LayerNorm(out_channels,bias=False)
        self.p = p

    def forward(self, x, pool_size=2):
        """
            Input shape: [b,L,chn]

            Out shape : [b,L//pool_size,chn]
        """
        x = x.tranpose(-1,-2)

        x = self.layer(x) # same as orig x shape
        x = F.avg_pool1d(x, kernel_size=pool_size,stride=pool_size)
        x = F.dropout(x,p=self.p)
    
        return x.transpose(-1,-2) # shape: [b,L//4,chn]


class net(nn.Module):
   
    def __init__(self,sequence_length,num_blocks,activation,lstm_option=False) -> None:
        super().__init__()
        self.sequence_length = sequence_length  
        #self.stft = STFT(sr=SR,n_fft=N_FFT,win_length=N_FFT,hop_length=HOP_LENGTH,trainable=True,fmin=20,fmax=20000,output_format="Complex")
        #self.istft = iSTFT(n_fft=N_FFT,n_iter=32,hop_length=HOP_LENGTH,win_length=N_FFT,trainable_kernels=True,trainable_window=False,fmin=20,fmax=20000,sr=SR,)
        self.stft = TorchSTFT()
        self.shape_spectrogram:tuple = self.calculate_spectrogram_shape(sequence_length)

        self.num_blocks = num_blocks
        self.activation_fn = get_activation_fn(activation)
        
        #self.enc_blocks = nn.ModuleList([]) 
        #self.enc_blocks.append(ConvBlock(in_channels=1,out_channels=64)) # b, L//2, 64
        #self.enc_blocks.append(ConvBlock(in_channels=64,out_channels=128)) # b, L//4, 128
        #self.enc_blocks.append(ConvBlock(in_channels=128,out_channels=256)) # b, L//8, 256

        self.dec_blocks = nn.ModuleList([])

        i = 0
        if lstm_option == 1:
            True if i % 2 == 0 else False
        
        for i in range(num_blocks):
            self.dec_blocks.append(TransformerBlock(embed_dim=self.shape_spectrogram[0],input_chn=self.shape_spectrogram[1],layer_index=i+1,lstm=lstm_option))
        # I checked for LSTM=True, and the performance was terrible(not even trainable)
        self.ln = nn.LayerNorm(self.shape_spectrogram[0])
        self.last_layer = nn.Sequential(

            
            nn.Linear(self.shape_spectrogram[0],self.shape_spectrogram[0]*4),
            nn.SiLU(),
            nn.Linear(self.shape_spectrogram[0]*4,self.shape_spectrogram[0]),
        )
        
        self.last2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.shape_spectrogram[1],self.shape_spectrogram[1]*4),
            nn.Dropout(p=0.1),
            nn.SiLU(),
            nn.Linear(self.shape_spectrogram[1]*4,self.shape_spectrogram[1]),
        )
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
            mag,angle = self.stft.transform(x) # batch, freq_bins, frames | batch, freq_bins, frames
            x = torch.cat((mag,angle),dim=1) # batch, 2 * freq_bins, frames
            x = x.transpose(-1,-2) # batch, frames[t], 2 * freq_bins
            
        for block in self.dec_blocks:
            x = block(x)
        '''
        
        '''
        x = x.transpose(-1,-2)
        x = self.ln(x) # for last-layer-of-TransformerBlock [unnormalized]
        x = self.last_layer(x).transpose(-1,-2)
        x = self.last2(x)

        spec = x[:,:,:N_FFT // 2 + 1]
        phase = x[:, :, N_FFT // 2 + 1:]
        return self.stft.inverse(spec,phase) # [1, L]
    



