import torch
from torch import einsum
import torch.nn as nn
from inspect import isfunction
from einops import rearrange,repeat
import math
import numpy as np

import torch.nn.functional
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in,dim_out * 2)
    def forward(self,x):
        x,gate = self.proj(x).chunk(2,dim=-1)
        return x * torch.nn.functional.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self,dim,dim_out=None,mult=4,glu=False,dropout=0.):
        super().__init__()
        inner_dim = int(dim*mult)
        dim_out = default(dim_out,dim)
        project_in = nn.Sequential(
            nn.Linear(dim,inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim,inner_dim)
        self.net = nn.Sequential(
            project_in,nn.Dropout(dropout),nn.Linear(inner_dim,dim_out)
        )

    def forward(self,x):
        return self.net(x)
    

class CrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim,query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k = nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v = nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,query_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self,x,context=None,mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context,x)
        k = self.to_k(context)
        v = self.to_v(context)

        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> (b h) n d',h=h).contiguous(),(q,k,v))
        sim = einsum('b i d , b j d -> b i j',q,k) * self.scale

        if exists(mask):
            mask = rearrange(mask,'b ... -> b (...)').contiguous()
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask,'b j -> (b h) () j',h=h)
            sim.masked_fill_(~mask,max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j , b j d -> b i d',attn,v)
        out = rearrange(out,'(b h) n d -> b n (h d)',h=h).contiguous()
        return self.to_out(out)

class PositionalEncoder(nn.Module):
    def __init__(self,d_model,seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pe = torch.zeros((seq_len,d_model))
        for pos in range(seq_len):
            for i in range(0,d_model-1,2):
                pe[pos,i] = math.sin(pos/(10000 ** ((2*i)/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000 **((2*(i+1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        if len(x.shape)>2:
            x = x.reshape(x.shape[0],x.shape[1],-1)
        seq_len = x.size(1)
        x = x + 1e-3 * self.pe[:,:seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_embedding, d_k, d_v, seq_len=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_embedding, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_embedding, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_embedding, n_head * d_v, bias=False)
        self.w_combine = nn.Linear(n_head * d_v, d_embedding, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([seq_len, d_embedding], eps=1e-6)

    def forward(self, q, k, v, mask=None):
        b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(b, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(b, len_v, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).contiguous().view(b, len_q, -1)
        z = self.dropout(self.w_combine(z))
        z = self.layer_norm(z + residual)  # Add & Norm
        return z, attn

class TemporalAttention(nn.Module):
    def __init__(self,n_head,d_embedding,d_k,d_v,seq_len=1,dropout=0.1):
        super().__init__()
        self.attention_layer = MultiHeadAttention(n_head,d_embedding,d_k,d_v,seq_len,dropout)
        self.positional_encoder = PositionalEncoder(6,seq_len)
    def forward(self,ref_img,hdmap,mask=None):
        x = np.concatenate((ref_img,hdmap),axis=3)
        n,h,w,c = x.shape[0],x.shape[2],x.shape[3],x.shape[4]

        x = rearrange(x,'b n h w c -> b c (n h w)').contiguous()
        x = self.positional_encoder(x)
        z,attn = self.attention_layer(x,x,x)
        z = rearrange(z,"b (h w) (n c) -> b n c h w",n=n,h=h,w=w,c=c).contiguous()

        return z

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,n_head,d_embedding,d_k,d_v,seq_len=1,dropout=0.1):
        super().__init__()
        self.attention_layer = MultiHeadAttention(n_head,d_embedding,d_k,d_v,seq_len,dropout)
    
    def forward(self,context,img,mask=None):
        z,attn = self.attention_layer(context,img,img)
        return z

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)
    




