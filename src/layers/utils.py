import torch
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from scipy.signal import get_window



class TorchSTFT(nn.Module):
    # from: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py#L456
    def __init__(self,config):
        super().__init__()
        self.filter_length = config['n_fft']
        self.hop_length = config['hop_length']
        self.win_length = config['win_len']

        self.window = torch.from_numpy(get_window('hann', config['win_len'], fftbins=True).astype(np.float32))

    def transform(self, input_data):
        if input_data.dim() == 3 and input_data.size(1) == 1:
            input_data.squeeze(1) 
        
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform) 
        # mag & phase. phase is being used but maybe tough for the model to learn, but essential for high quality.

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

    
class PositionwiseFeedForward(nn.Sequential):

    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.2):
        super().__init__(
            nn.Linear(dims, dims * rate),
            nn.SiLU(),
            nn.AlphaDropout(dropout),
            nn.Linear(dims * rate, dims))