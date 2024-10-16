import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func #pip install flash-attn --no-build-isolation
from rotary_embedding_torch import RotaryEmbedding
from util_net import RMSNorm,PositionwiseFeedForward

class MultiheadFlashDiff(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_flashdiff_1.py
    def __init__(
        self,
        embed_dim, # freq_bins for orig
        depth, # layer num. [1 to N layer]
        num_heads, # best for 2, as one can focus on low_freq and one can focus on
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.25
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.ln = RMSNorm(2 * self.head_dim, eps=1e-8,bias=True)
        self.rotary = RotaryEmbedding(dim=self.head_dim,seq_before_head_dim=True,freqs_for='lang',interpolate_factor=1,cache_if_possible=True,use_xpos=True)
        # seq_before_head_dim = False sets seq_dim as -2, not -3.
        
    def forward(
        self,
        x,
    ):  
        b, seq_len, embed_dim = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, seq_len, 2 * self.num_heads, self.head_dim)
        k = k.view(b, seq_len, 2 * self.num_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_heads, 2 * self.head_dim)

        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)

        q = q.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1] # same as q[:,:,:,0,:].squeeze(-1). it's correct!
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn1 = flash_attn_func(q1, k1, v, causal=True)
        attn2 = flash_attn_func(q2, k2, v, causal=True)
        
        
        attn = self.ln(attn1 - self.get_lambda() * attn2)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(b, seq_len, self.embed_dim)
        attn = self.out_proj(attn)

        return attn
    
    def get_lambda(self,):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        return lambda_full





class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, depth,num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads


        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.ln3 = RMSNorm(embed_dim)

        self.attn = MultiheadFlashDiff(embed_dim=embed_dim,depth=depth,num_heads=num_heads)
        self.ff = PositionwiseFeedForward(dims=embed_dim,)
        self.out = nn.Linear(embed_dim,embed_dim,bias=False)

    
    def forward(self, x):
        '''
        Get x: [batch,seq,embed_dim]
        '''

        x = self.ln1(x)
        x = self.ln2(self.attn(x) + x)
        x = self.ln3(self.ff(x) + x)
        x = self.out(x)
        return x
