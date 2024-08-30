from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn,einsum
from einops import rearrange,repeat
from ldm.models.attention import PositionalEncoder
from ldm.modules.diffusionmodules.util import checkpoint

def exists(val):
    return val is not None

def uniq(arr):
    return {el:True for el in arr}.keys()

def default(val,d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_meg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std,std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in,dim_out*2)

    def forward(self,x):
        x,gate = self.proj(x).chunk(2,dim=-1)
        return x * F.gelu(gate)
    
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
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim,dim_out)
        )
    def forward(self,x):
        return self.net(x)
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,num_channels=in_channels,eps=1e-6,affine=True)

class LinearAttention(nn.Module):
    def __init__(self,dim,heads=4,dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head*heads
        self.to_qkv = nn.Conv2d(dim,hidden_dim*3,1,bias=False)
        self.to_out = nn.Conv2d(hidden_dim,dim,1)

    def forward(self,x):
        b,c,h,w = x.shape
        qkv = self.to_qkv(x)
        q,k,v = rearrange(qkv,'b (qkv heads c) h w -> qkv b heads c (h w)',self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde',k,v)
        out = torch.einsum('bhde,bhdn->bhen',context,q)
        out = rearrange(out,'b heads c (h w) -> b (heads c) h w',heads=self.heads,h=h,w=w)
        return self.to_out(out)
    
class TemporalAttention(nn.Module):
    def __init__(self,
                channels,
                num_heads=1,
                num_head_channels=-1,
                use_checkpoint=False,
                use_new_attention_order=False,
                movie_len=None):
        super().__init__()
        # self.attention_block = AttentionBlock(channels,num_heads,num_head_channels,use_checkpoint,use_new_attention_order)
        self.temporal_block = CrossAttention(channels,heads=num_heads,dim_head=num_head_channels)
        self.movie_len = movie_len
        self.positional_encoder = PositionalEncoder(channels,movie_len)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self,x,_=None):
        bn = x.shape[0]
        batch_size = bn // self.movie_len
        x = rearrange(x,'(b n) c h w -> (b h w) n c',b=batch_size,n=self.movie_len).contiguous()
        x = x + self.positional_encoder(x)
        out = self.temporal_block(self.norm(x)) + x
        return out

class SpatialSelfAttention(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
    def forward(self,x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        #compute attention
        b,c,h,w = q.shape
        q = rearrange(q,'b c h w -> b (h w) c').contiguous()
        k = rearrange(k,'b c h w -> b c (h w)').contiguous()
        w_ = torch.einsum('bij,bjk->bik',q,k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)

        #attend to values
        v = rearrange(v,'b c h w -> b c (h w)').contiguous()
        w_ = rearrange(w_,'b i j -> b j i').contiguous()
        h_ = torch.einsum('bij,bjk->bik',v,w_)
        h_ = rearrange(h_,'b c (h w) -> b c h w',h=h).contiguous()
        h_ = self.proj_out(h_)
        return x+h_
    
class Lidar_Image_CrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim,heads=8,dim_head=64,dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim,query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k = nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v = nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,query_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x,context=None,mask=None):
        h = self.heads
        b = x.shape[0]
        q = self.to_q(x)
        context = default(context,x)

        key_0,key_1 = torch.chunk(x,chunks=2,dim=0)
        value_0,value_1 = torch.chunk(x,chunks=2,dim=0)
        key_L,key_I = torch.cat([key_1,key_0],dim=-1),torch.cat([key_0,key_1],dim=-1)
        value_L,value_I = torch.cat([value_1,value_0],dim=-1),torch.cat([value_0,value_1],dim=-1)
        k = torch.cat([key_I,key_L],dim=0)
        v = torch.cat([value_I,value_L],dim=0)
        k = self.to_k(k)
        v = self.to_v(v)
        q,k,v = map(lambda t: rearrange(t,'b n (h d) -> (b h) n d',b=b,h=h,d=self.dim_head,n=t.shape[1]).contiguous(),(q,k,v))
        sim = einsum('b i d, b j d -> b i j',q,k) * self.scale
        if exists(mask):
            mask = rearrange(mask,'b ... -> b (...)').contiguous()
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d',attn,v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        return self.to_out(out)



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        b = x.shape[0]
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', b=b,h=h,d=self.dim_head,n=t.shape[1]).contiguous(), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)').contiguous()
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        return self.to_out(out)
    
class GatedSelfAttention(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)
        self.gated_attention_block = CrossAttention(query_dim,heads=n_heads,dim_head=d_head)
        
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs=None):

        N_visual = x.shape[1]
        if not objs is None:
            objs = self.linear(objs)
            h = self.norm1(torch.cat([x,objs],dim=1))
        else:
            h = self.norm1(x)
        # h = rearrange(h,'b n c -> b c n')
        h = self.scale * torch.tanh(self.alpha_attn) * self.gated_attention_block(h)
        # h = rearrange(h,'b n c -> b c n')
        x = x + h[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,movie_len=None,height=None,width=None,obj_dims=None,use_additional_attn=False):
        super().__init__()
        self.attn1 = GatedSelfAttention(query_dim=dim,context_dim=obj_dims, n_heads=n_heads, d_head=d_head)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.use_additional_attn = use_additional_attn
        if self.use_additional_attn:
            self.attn3 = Lidar_Image_CrossAttention(query_dim=dim,context_dim=2*dim,heads=n_heads,dim_head=d_head)
            self.norm4 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.movie_len = movie_len
        self.height = height
        self.width = width

    def forward(self, x, objs=None,context=None):
        return checkpoint(self._forward, (x, objs,context), self.parameters(), self.checkpoint)

    def _forward(self, x, objs=None,context=None):
        # with temporal blocks
        bhw = x.shape[0]
        batch_size = bhw // self.height // self.width
        x = rearrange(x,'(b h w) n c -> (b n) (h w) c',b=batch_size,h=self.height,w=self.width,n=self.movie_len).contiguous()
        if objs is None:
            x = self.attn1(self.norm1(x)) + x
        else:
            x = self.attn1(self.norm1(x),objs) + x
        
        x = self.attn2(self.norm2(x), context=context) + x
        
        if self.use_additional_attn:
            x = self.attn3(self.norm4(x)) + x

        x = self.ff(self.norm3(x)) + x
        
        # without temporal blocks
        # bn = x.shape[0]
        # batch_size = bn / self.movie_len
        # x = rearrange(x,'bn c h w -> bn (h w) c',h=self.height,w=self.width)
        # x = self.attn1(self.norm1(x),objs) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self,in_channels,n_heads,d_head,
                 depth=1,dropout=0.,context_dim=None,movie_len=None,height=None,width=None,obj_dims=None,use_attn_additional=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.movie_len = movie_len
        self.height = height
        self.width = width
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.transformer_blocks = []
        # self.temporal_blocks = []
        # for d in range(depth):
        #     self.temporal_blocks.append(TemporalAttention(inner_dim,n_heads,d_head,movie_len=movie_len))
        #     self.transformer_blocks.append(BasicTransformerBlock(inner_dim,n_heads,d_head,dropout=dropout,context_dim=context_dim,obj_dims=obj_dims,batch_size=batch_size,movie_len=movie_len,height=height,width=width))    
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(inner_dim,n_heads,d_head,dropout=dropout,context_dim=context_dim,obj_dims=obj_dims,movie_len=movie_len,height=height,width=width,use_additional_attn=use_attn_additional) for d in range(depth)
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalAttention(inner_dim,n_heads,d_head,movie_len=movie_len) for d in range(depth)
        )
        # self.transformer_blocks = nn.ModuleList(
        #     [item
        #      for d in range(depth) for item in [TemporalAttention(inner_dim,n_heads,d_head,movie_len=movie_len),
        #      BasicTransformerBlock(inner_dim,n_heads,d_head,dropout=dropout,context_dim=context_dim,obj_dims=obj_dims,batch_size=batch_size,movie_len=movie_len,height=height,width=width)]]
        # )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        
    def forward(self,x,boxes_emb=None,text_emb=None):
        # b,c,h,w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        # x = rearrange(x,'b c h w -> b (h w) c')
        for d in range(len(self.transformer_blocks)):
            x = self.temporal_blocks[d](x)
            if boxes_emb is None:
                x = self.transformer_blocks[d](x,context=text_emb)
            else:
                x = self.transformer_blocks[d](x,boxes_emb,text_emb)
        # for block in self.transformer_blocks:
        #     x = block(x,boxes_emb,text_emb)
        # x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w)
        x = rearrange(x,'bn (h w) c -> bn c h w',h=self.height,w=self.width).contiguous()
        x = self.proj_out(x)
        return x + x_in

