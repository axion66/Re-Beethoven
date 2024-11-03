import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from flash_attn import flash_attn_func #pip install flash-attn --no-build-isolation
from rotary_embedding_torch import RotaryEmbedding
from layers.tools.utils import *
try:
    from flash_attn import flash_attn_func
except:
    print("Flash attention not supported. try revising the code")
from tools.norms import RMSNorm,LayerNorm








class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        norm_fn=None,
        activation_fn=None,
        p=0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads


        self.ln1 = norm_fn(embed_dim) if exists(norm_fn) else LayerNorm(embed_dim)
        self.ln2 = norm_fn(embed_dim) if exists(norm_fn) else LayerNorm(embed_dim)

        self.attn = DiffMHAFlash(embed_dim=embed_dim, 
                                       depth=depth,
                                       num_heads=num_heads)
        self.ff = PositionwiseFeedForward(dims=embed_dim,
                                          activation=activation_fn if exists(activation_fn) else nn.SiLU(),
                                          rate=4,
                                          dropout=p
                                          )

    def forward(self, x,sigmas):
        '''
        Get x: [batch,seq,embed_dim]
        Get sigmas: [batch,head_dim]
        '''

        x = self.ln1(self.attn(x,sigmas) + x)
        x = self.ln2(self.ff(x) + x)

        return x


class DiffMHAFlash(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_flashdiff_1.py
    # Differential Attention for precise attention score


    def __init__(
        self,
        embed_dim, # freq_bins for orig
        depth, # layer num. [1 to N layer]
        num_heads, # best for 2?, as one can focus on low_freq and one can focus on high freq (like RoPE)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads // 2
        
        self.qkv = Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = Linear(embed_dim, embed_dim, bias=False)
        self.reset_lambda(depth)
        
        self.ln = RMSNorm(2 * self.head_dim, eps=1e-8,bias=True)
        self.ln_qkv2 = RMSNorm(self.head_dim, eps=1e-8,bias=True)
        self.rotary = RotaryEmbedding(dim=self.head_dim//2, 
                                      seq_before_head_dim=True,
                                      freqs_for='lang', 
                                      interpolate_factor=1,
                                      cache_if_possible=True, 
                                      use_xpos=False)
    
    def reset_lambda(self,depth):
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

    def lmd(self,q):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        return lambda_full
    
    def forward(
        self,
        x,
        sigmas
    ):  
        b, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # QKV Split
        q,k,v = torch.chunk(self.qkv(x),dim=-1,chunks=3)
        q = q.view(b, seq_len, 2 * self.num_heads, self.head_dim)   # batch, seq_len, 2 * n, h
        k = k.view(b, seq_len, 2 * self.num_heads, self.head_dim)   # batch, seq_len, 2 * n, h
        v = v.view(b, seq_len, self.num_heads, 2 * self.head_dim)   # batch, seq_len, n,  2 * h
        
        # Apply RoPE            - (Think This's best option for positional embedding)
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)

        # Reshape Q, K, Sigmas
        q = q.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1] 
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        sigmas = sigmas.unsqueeze(1).unsqueeze(2)   # batch,1,1,head_dim
        
        # Differential Attention
        attn1 = flash_attn_func(q1, k1, v, casual=True)
        attn2 = flash_attn_func(self.ln_qkv2(q2 + sigmas), self.ln_qkv2(k2 + sigmas), v, casual=True)
        attn = self.ln(attn1 - self.lmd(q) * attn2) * (1 - self.lambda_init)

        # Reshape and Linear projection
        attn = attn.reshape(b, seq_len, self.embed_dim)
        return self.out(attn)
    
    

    '''
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute the scaled dot-product attention.
        Created By GPT for debugging in non-gpu(non-flashattn) purpose only
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)

        Returns:
            output: Attention output tensor of shape (batch_size, seq_len, num_heads, head_dim)
            attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Calculate the attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, seq_len, num_heads, seq_len)
        scaling_factor = self.head_dim ** 0.25
        scores = scores / scaling_factor

        # Apply the mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute the attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len, num_heads, seq_len)

        # Compute the attention output
        output = torch.matmul(attention_weights, v)  # Shape: (batch_size, seq_len, num_heads, head_dim)

        return output
    '''

