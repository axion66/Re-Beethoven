import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from layers.tools.utils import *
from layers.tools.activations import *
from layers.tools.norms import *
from abc import abstractmethod
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from flash_attn import flash_attn_func
    FLASH_ON = True
except:
    print("Flash attention not supported. try revising the code")
    FLASH_ON = False
    
    

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class DiTBlock(TimestepBlock):
    def __init__(
        self,
        num_blocks,
        latents_dim,
        embed_dim,
        num_heads,
        norm_fn=None,
        p=0.1,
    ):
        super().__init__()
        self.latents_dim = latents_dim
        self.embed_dim = embed_dim

        


        #   initial layer (latent_dim -> embed_dim)
        self.latents2embed = nn.Sequential(
            Linear(self.latents_dim, self.embed_dim),
            Rearrange('b l c -> b c l'),
            SnakeBeta(in_features = self.embed_dim),
            Rearrange('b c l -> b l c'),
            Linear(self.embed_dim, self.embed_dim),
        )

        #   dit
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                              embed_dim = self.embed_dim,
                              depth = i + 1,
                              num_heads = num_heads,
                              norm_fn = norm_fn,
                              p = p,
                            ) 
                for i in range(num_blocks)
            ]
        )

        #   final layer (embed_dim -> latent_dim)
        self.final_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps = 1e-6)
        self.adaLN_modulation = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias = True)
        self.final_linear = nn.Linear(self.embed_dim, self.latents_dim, bias = True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    
    def forward(self, x, emb):
        x = self.latents2embed(x)

        for block in self.transformer:
            x = block(x, emb)


        scale, shift = self.adaLN_modulation(emb).unsqueeze(1).chunk(2, dim = -1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.final_linear(x)
        return x

class TransformerBlock(TimestepBlock):

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        norm_fn=None,
        p=0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads


        self.norm_1 = norm_fn(embed_dim) if exists(norm_fn) else LayerNorm(embed_dim)
        self.norm_2 = norm_fn(embed_dim) if exists(norm_fn) else LayerNorm(embed_dim)

        self.attn = DifferentialAttention(
            embed_dim=embed_dim, 
            depth=depth,
            num_heads=num_heads     
        )
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*4),
            Rearrange('b l c -> b c l'),
            SnakeBeta(in_features = embed_dim*4),
            Rearrange('b c l -> b l c'),
            nn.Linear(embed_dim*4,embed_dim)
        )
        
        self.set_adaLN(embed_dim)  
        
    def set_adaLN(self, embed_dim):
        self.adaln = nn.Linear(embed_dim, embed_dim * 6, bias=True)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x, emb):
        '''
        Get x: [batch,seq,embed_dim]
        Get sigmas: [batch, embed_dim]
        
        
            Pre Normalization, but no Post Normalization being applied.
        '''
        assert emb.ndim == 2
        scale_msa, shift_msa, scale_mlp, shift_mlp, gate_msa, gate_mlp = self.adaln(emb).unsqueeze(1).chunk(6, dim = -1)
        

        res = x
        x = self.norm_1(x)
        x = x * (1 + scale_msa) + shift_msa
        x = self.attn(x)
        x = x * torch.sigmoid(1 - gate_msa) 
        x = x + res
        
        res = x
        x = self.norm_2(x)
        x = x * (1 + scale_mlp) + shift_mlp
        x = self.ff(x)
        x = x * torch.sigmoid(1 - gate_mlp)
        x = x + res
          
        return x



class DifferentialAttention(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_flashdiff_1.py
    # Differential Attention for precise attention score


    def __init__(
        self,
        embed_dim, # freq_bins for orig
        depth, # layer num. [1 to N layer]
        num_heads, # best for 2?, as one can focus on low_freq and one can focus on high freq (like RoPE)
        dim_out=None    # Optional dim_out. By default, it's same as embed_dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads // 2
        
        self.qkv = Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = Linear(embed_dim, dim_out if dim_out is not None else embed_dim, bias=False)
        self.reset_lambda(depth)
        
        self.ln = RMSNorm(2 * self.head_dim, eps=1e-8,bias=True)
        self.ln_qkv2 = RMSNorm(self.head_dim, eps=1e-8,bias=True)
        self.rotary = RotaryEmbedding(dim=self.head_dim//2, 
                                      seq_before_head_dim=True,
                                      freqs_for='lang', 
                                      interpolate_factor=1,
                                      use_xpos=False)

    def reset_lambda(self, depth):
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
        x
        ):   
        # NOV 24: remove adding sigmas, 
        # but rather use adaLN-zero in transformer level.
        # note that w/ correct implementation of DDPM, regular flash-attn worked w/o any problems.
        b, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        q,k,v = torch.chunk(self.qkv(x),dim=-1,chunks=3)
        q = q.view(b, seq_len, 2 * self.num_heads, self.head_dim)   # batch, seq_len, 2 * n, h
        k = k.view(b, seq_len, 2 * self.num_heads, self.head_dim)   # batch, seq_len, 2 * n, h
        v = v.view(b, seq_len, self.num_heads, 2, self.head_dim)    # batch, seq_len, n,  2 * h
        #v = v.view(b, seq_len, 2 * self.num_heads, self.head_dim)
        
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)
        '''if FLASH_ON:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            attn = flash_attn_func(q,k,v,causal=True)
            attn = rearrange(attn, "b l n h -> b l (n h)").float()
            return self.out(attn)'''
        
        q = q.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(b, seq_len, self.num_heads, 2, self.head_dim)
        
        if FLASH_ON:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1] 
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]
        
       
        if FLASH_ON:            
            # First attention pair
            attn11 = flash_attn_func(q1, k1, v1, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, causal=True)
            
            # Second attention pair
            attn21 = flash_attn_func(q2, k2, v1, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, causal=True)
        else:
            attn11 = self.qkv_attn(q1, k1, v1)
            attn12 = self.qkv_attn(q1, k1, v2)
            
            attn21 = self.qkv_attn(q2, k2, v1)
            attn22 = self.qkv_attn(q2, k2, v2)
        
        a1 = torch.cat((attn11,attn12), dim=-1)
        a2 = torch.cat((attn21,attn22), dim=-1)
        a = self.ln(a1 - self.lmd(q) * a2) * (1 - self.lambda_init)
        a  = rearrange(a, "b l n h -> b l (n h)")
        return self.out(a)
    
    

    
    def qkv_attn(self, q, k, v):
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
        scaling_factor = self.head_dim ** -0.5
        scores = scores * scaling_factor

      
        # Compute the attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len, num_heads, seq_len)

        # Compute the attention output
        output = torch.matmul(attention_weights, v)  # Shape: (batch_size, seq_len, num_heads, head_dim)

        return output
    
