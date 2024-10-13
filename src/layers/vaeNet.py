from layers.minLSTM_block import MinLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import rearrange
from scipy.signal import get_window

from nnAudio.features import MelSpectrogram,STFT,iSTFT# instead, use TorchSTFT class.
from nnAudio.Spectrogram import Griffin_Lim

N_FFT = 2048
HOP_LENGTH = 1024
CHUNK_LENGTH = 30
N_MELS = 64
SR = 8000

class TorchSTFT(nn.Module):
    # from: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py#L456 [MIT License]
    def __init__(self):
        super().__init__()
        self.filter_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.win_length = N_FFT
        self.window = torch.from_numpy(get_window('hann', N_FFT, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform) # mag & phase. phase is being used but maybe tough for the model to learn, but essential.

    def inverse(self, magnitude, phase):
        # TODO: the length should be divisible by hop_length in order for perfect-length reconstruction.
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j), # waveform = mag * e^phase
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
        
    
class moving_avg(nn.Module):
    # maybe trying EMA or WMA also good idea, but computation should be handled on the CPU.
    def __init__(self, kernel_size, stride):
        #kernel_size recommend [5,20,40,120]
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        '''
            x: [batch,len,chn]
        '''
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class decompose(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 
        
        
class PositionwiseFeedForward(nn.Sequential):
    # from: https://github.com/affjljoo3581/GPT2/tree/master
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims))
        
class diffAttn(nn.Module):
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

        
        self.ln_attn = RMSNorm(embed_dim) # RMS is center-variant
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
        self.ln_ff = RMSNorm(embed_dim)
        self.ff = PositionwiseFeedForward(dims=embed_dim,)


    def set_lambda(self, layer_index):
        return 0.8 - 0.6 * torch.exp(-0.3 * layer_index)

    def forward(self, x):
        '''
        Get x: [batch,seq,chn]
        '''
        a = self.ln_attn(x.transpose(-1,-2)).transpose(-1,-2) # post-layernorm so that we can use bias in the linaer/lstm net.
        trend,seasonal = self.decomposition(a)
        a = self.attn(trend,seasonal,a)
        x = x + a
        x = x + self.ff(self.ln_ff(x).transpose(-1,-2)).transpose(-1,-2)
        return x
    
    def attn(self, trend,seasonal,a):
        
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




class net(nn.Module):
   
    def __init__(self,sequence_length,num_blocks,activation) -> None:
        super().__init__()
        self.sequence_length = sequence_length  
        self.stft = STFT(sr=SR,n_fft=N_FFT,win_length=N_FFT,hop_length=HOP_LENGTH,trainable=True,fmin=20,fmax=20000,output_format="Complex")

        self.shape_spectrogram:tuple = self.calculate_spectrogram_shape(sequence_length)

        self.num_blocks = num_blocks
        self.activation_fn = get_activation_fn(activation)
        
        self.enc_blocks = nn.ModuleList([])
        self.dec_blocks = nn.ModuleList([])

        self.enc_blocks.append(diffAttn(embed_dim=self.shape_spectrogram[1],input_chn=self.shape_spectrogram[0],layer_index=1))
        for i in range(num_blocks-1):
            self.enc_blocks.append(diffAttn(embed_dim=self.shape_spectrogram[1],input_chn=512,layer_index=i+2))
        
        for i in range(num_blocks):
            self.dec_blocks.append(diffAttn(embed_dim=self.shape_spectrogram[1],input_chn=512,layer_index=i+1))
        
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(self.shape_spectrogram[1] * 512, sequence_length // 4),
            self.activation_fn(),
            nn.Linear(sequence_length//4,sequence_length)
        )

        self.istft = iSTFT(n_fft=N_FFT,n_iter=32,hop_length=HOP_LENGTH,win_length=N_FFT,trainable_kernels=True,trainable_window=False,fmin=20,fmax=20000,sr=SR,)

    def calculate_spectrogram_shape(self,sequence_length):
        # Calculate the number of frames (time dimension)
        with torch.no_grad():
            out =self.spectrogram(torch.ones((1,1,sequence_length)))
            return out.shape[1],out.shape[2]

    def forward(self,x): 
        '''
            input x: [batch,1,seq]
        '''
        x = self.stft(x) # batch,n_mels,num_frames
        print(x.shape)
        x = ( x - x.mean(dim=-1,keepdim=True)) / x.std(dim=-1,keepdim=True)
        x = x.transpose(-1,-2) # batch,seq_len,n_mels
        

        for block in self.enc_blocks:
            x = block(x)

        
        for block in self.dec_blocks:
            x = block(x)


        return self.istft(x)