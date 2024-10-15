import torch
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from scipy.signal import get_window

# config
SR = 8000
N_FFT = 512
WIN_LEN = 512
HOP_LENGTH = 256
CHUNK_LENGTH = 30
N_MELS = 64
MIN=1e-7
MAX=2e+5

class TorchSTFT(nn.Module):
    # from: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py#L456
    def __init__(self):
        super().__init__()
        self.filter_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.win_length = WIN_LEN
        self.window = torch.from_numpy(get_window('hann', WIN_LEN, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        # input data has to be 1D or 2D [L] or [Batch, L]. no [Batch,1,L]
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform) # mag & phase. phase is being used but maybe tough for the model to learn, but essential for high quality.

    def inverse(self, magnitude, phase):

        magnitude = magnitude.transpose(-1,-2)
        phase = phase.transpose(-1,-2)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j), # waveform = mag * e^phase
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform
    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)

        self.magnitude = self.magnitude.transpose(-1,-2)
        self.phase = self.phase.transpose(-1,-2)
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
            nn.AlphaDropout(dropout),
            nn.Linear(dims * rate, dims))