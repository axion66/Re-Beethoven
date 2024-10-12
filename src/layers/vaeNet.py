from layers.minLSTM_block import MinLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import rearrange
from nnAudio.features import MelSpectrogram


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

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 
        
        
        
        

class diffAttn(nn.Module):
    '''
        a weird combination of iTransformer, DLinear, DiffTransformer, and MinLSTM. 
        
    '''
    def __init__(self,embed_dim,input_chn,layer_index):
        '''
            mixed diff attention with minLSTM and trend,seasonal decomposition.
        
            turn [batch,seq,chn] -> [batch,seq{->embed_dim},512]
        '''
        super().__init__()
        self.embedding = nn.Linear(input_chn,512)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.decomposition = series_decomp(3)
       
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.25
        self.q12 = MinLSTM(input_size=512,hidden_size=512*2)
        self.k12 = MinLSTM(input_size=512,hidden_size=512*2)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_init = self.set_lambda(torch.Tensor([layer_index])) if layer_index is not None else 0.8
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5, elementwise_affine=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def set_lambda(self,layer_index):
        return 0.8 - 0.6* torch.exp(-0.3 * (layer_index))


    def forward(self,x):
        '''
        get x: [batch,seq,chn]
        
        '''

        x = self.embedding(x) # [batch,seq,512]
        return self.attn(x)
    
    def attn(self,x):
        trend,seasonal = self.decomposition(x)
        # trend: [batch,seq,512], seasonal: [batch,seq,512] where 512 = d, seq = s
        q1,q2 = torch.chunk(self.q12(trend),chunks=2,dim=-1) # q1 = b,s,d q2= b,s,d
        k1,k2 = torch.chunk(self.k12(seasonal),chunks=2,dim=-1) # k1 = b,s,d k2 = b,s,d
        v = self.v(x.transpose(-1,-2))  # v = b,d,s

        attn1 = torch.bmm(q1.transpose(-1,-2),k1) * self.scale # batch,d,d
        attn2 = torch.bmm(q2.transpose(-1,-2),k2) * self.scale # batch,d,d

        # from: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py#L23
        
        difference = F.softmax(attn1,dim=-1) - self.get_lambda() * F.softmax(attn2,dim=-1) # b,d,d
        attn =  torch.bmm(difference,v) # b,d,s
        attn = self.norm(attn) # b,d,s
        attn *= (1 - self.lambda_init) # b,s,d
        return self.out_proj(attn).transpose(-1,-2) # b,s,d

    def get_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        return lambda_full
    
class VAE_temporal(nn.Module):

    def __init__(self,num_blocks,activation) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.activation_fn = get_activation_fn(activation)
        
        self.enc_blocks = nn.ModuleList([])
        self.dec_blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            
            pass
    def encode(self,):
        pass

    def decode(self,):
        pass


    def forward(self,):
        pass

    def kl(self,):
        pass

N_FFT = 2048
HOP_LENGTH = 1024
CHUNK_LENGTH = 30
N_MELS = 64
SR = 8000
class net(nn.Module):
   
    def __init__(self,sequence_length,num_blocks,activation) -> None:
        super().__init__()
        self.sequence_length = sequence_length  
        self.spectrogram = MelSpectrogram(sr=SR,n_fft=N_FFT,win_length=N_FFT,n_mels=N_MELS,hop_length=HOP_LENGTH,trainable_mel=True,trainable_STFT=True,htk=True,fmin=1e-7)

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
    def calculate_spectrogram_shape(self,sequence_length):
        # Calculate the number of frames (time dimension)
        with torch.no_grad():
            out =self.spectrogram(torch.ones((1,1,sequence_length)))
            return out.shape[1],out.shape[2]

    def forward(self,x): 
        '''
            input x: [batch,1,seq]
        '''
        x = self.spectrogram(x) # batch,n_mels,num_frames
        x = x.transpose(-1,-2) # batch,seq_len,n_mels

        for block in self.enc_blocks:
            x = block(x)

        
        for block in self.dec_blocks:
            x = block(x)


        return self.proj(x)# b,seq