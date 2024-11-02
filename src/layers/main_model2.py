import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.tools.audios import RevSTFT
from layers.tools.utils import RMSNorm, PositionwiseFeedForward, Linear

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = Linear(dim, inner_dim * 3, bias=False)
        self.to_out = Linear(inner_dim, dim)
        
    def forward(self, x):
        # x shape: [batch, frames, channels]
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)  # [batch, frames, channels]
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = Linear(time_emb_dim, out_channels) if time_emb_dim else None
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = RMSNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = RMSNorm(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x, time_emb=None):
        # Input shape: [batch, frames, channels]
        # Need: [batch, channels, frames] for Conv1d
        h = x.transpose(-1,-2)  # [batch, channels, frames]
        
        h = self.conv1(h)
        h = h.transpose(-1,-2)  # [batch, frames, channels] for norm
        h = self.norm1(h)
        h = self.act(h)
        
        if self.time_mlp and time_emb is not None:
            time_emb = self.act(self.time_mlp(time_emb))
            h = h + time_emb.unsqueeze(1)
            
        h = h.transpose(-1,-2)  # [batch, channels, frames]
        h = self.conv2(h)
        h = h.transpose(-1,-2)  # [batch, frames, channels]
        h = self.norm2(h)
        h = self.act(h)
        
        return h 
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(dim, heads)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: [batch, frames, channels]
        x = x + self.mha(self.norm(x))
        return x  # [batch, frames, channels]
class UNetWithMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stft = RevSTFT(config)
        
        # Calculate input dimensions
        self.seq_len, self.embed_dim = self.calculate_spectrogram_shape(config['seq_len'])
        self.sequence_len = self.seq_len
        self.time_mlp = nn.Sequential(
            FourierFeatures(1, self.embed_dim),
            Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            Linear(self.embed_dim, self.embed_dim)
        )
        time_dim = self.embed_dim
        self.sequence_length = self.seq_len
        # Encoder
        self.enc1 = ConvBlock(self.embed_dim, 64, time_dim)
        self.enc2 = ConvBlock(64, 128, time_dim)
        self.enc3 = ConvBlock(128, 256, time_dim)
        self.enc4 = ConvBlock(256, 512, time_dim)
        
        # Middle
        self.mid_attn = AttentionBlock(512)
        self.mid_block1 = ConvBlock(512, 512, time_dim)
        self.mid_block2 = ConvBlock(512, 512, time_dim)
        
        # Decoder
        self.dec4 = ConvBlock(1024, 256, time_dim)  # 512 + 512 = 1024
        self.dec3 = ConvBlock(512, 128, time_dim)   # 256 + 256 = 512
        self.dec2 = ConvBlock(256, 64, time_dim)    # 128 + 128 = 256
        self.dec1 = ConvBlock(128, self.embed_dim, time_dim)  # 64 + 64 = 128
        
        # Attention layers
        self.attn1 = AttentionBlock(64)
        self.attn2 = AttentionBlock(128)
        self.attn3 = AttentionBlock(256)

        self.reshaper = nn.Linear(4096,4097)
    def calculate_spectrogram_shape(self, sequence_length):
        with torch.no_grad():
            out, out2 = self.stft.transform(torch.ones((1, sequence_length)))
            ff = torch.cat((out, out2), dim=1)
            ff = ff.transpose(1, 2)  # [batch, frames, channels]
            return ff.shape[1], ff.shape[2]
    
    def forward(self, x, sigmas):
        # STFT transform
        mag, angle = self.stft.transform(x)
        x = torch.cat((mag, angle), dim=1)  # [batch, channels, frames]
        x = x.transpose(1, 2)  # [batch, frames, channels]
        
        # Time embedding
        t = self.time_mlp(sigmas)
        
        # Encoder path
        e1 = self.enc1(x, t)          # [batch, frames, 64]
        e1 = self.attn1(e1)
        
        e1_pool = e1.permute(0, 2, 1)  # [batch, 64, frames]
        e1_pool = F.avg_pool1d(e1_pool, 2)
        e1_pool = e1_pool.permute(0, 2, 1)  # [batch, frames/2, 64]
        
        e2 = self.enc2(e1_pool, t)    # [batch, frames/2, 128]
        e2 = self.attn2(e2)
        
        e2_pool = e2.permute(0, 2, 1)
        e2_pool = F.avg_pool1d(e2_pool, 2)
        e2_pool = e2_pool.permute(0, 2, 1)
        
        e3 = self.enc3(e2_pool, t)    # [batch, frames/4, 256]
        e3 = self.attn3(e3)
        
        e3_pool = e3.permute(0, 2, 1)
        e3_pool = F.avg_pool1d(e3_pool, 2)
        e3_pool = e3_pool.permute(0, 2, 1)
        
        e4 = self.enc4(e3_pool, t)    # [batch, frames/8, 512]
        
        # Middle
        x = self.mid_block1(e4, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Upsample x
        #x_up = x.permute(0, 2, 1)  # [batch, 512, frames/8]
        #x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        #x_up = x_up.permute(0, 2, 1)  # [batch, frames/4, 512]
        #print(x_up.shape)
        #print(e4.shape)
    
        x = torch.cat([x, e4], dim=2)  
        x = self.dec4(x, t)
        
        x_up = x.permute(0, 2, 1)
        x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        x_up = x_up.permute(0, 2, 1)
        
        x = torch.cat([x_up, e3], dim=2)
        x = self.dec3(x, t)
        
        x_up = x.permute(0, 2, 1)
        x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        x_up = x_up.permute(0, 2, 1)
        
        x = torch.cat([x_up, e2], dim=2)
        x = self.dec2(x, t)
        
        x_up = x.permute(0, 2, 1)
        x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        x_up = x_up.permute(0, 2, 1)
        
        x = torch.cat([x_up, e1[:,1:,:]], dim=2)
        x = self.dec1(x, t)

        x = x.transpose(-1,-2)
        x = self.reshaper(x)
        x = x.transpose(-1,-2)
        
        spec = x[:, :, :self.config['n_fft'] // 2 + 1]
        phase = x[:, :, self.config['n_fft'] // 2 + 1:]
        

        # Transpose back for STFT inverse
        #spec = spec.transpose(1, 2)
        #phase = phase.transpose(1, 2)
        
        return self.stft.inverse(spec, phase)