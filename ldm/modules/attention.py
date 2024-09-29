from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn,einsum
from einops import rearrange,repeat
from ldm.models.attention import PositionalEncoder
# from ldm.modules.diffusionmodules.util import checkpoint
from torch.utils.checkpoint import checkpoint
from ldm.modules.diffusionmodules.util import linear,timestep_embedding,AlphaBlender
from typing import Optional,Any
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True
        },
        None: {
            "enable_math": True,
            "enable_flash": True,
            "enable_mem_efficient": True
        }
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = dict()
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. "
        f"In fact, you are using PyTorch {torch.__version__}. You might want to consider upgrading"
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module `xformers`, processing without it")

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
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,movie_len=None,height=None,width=None,obj_dims=None,use_additional_attn=False,bev_dims=None):
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
        return checkpoint(self._forward, x, objs,context,use_reentrant=False)

    def _forward(self, x, objs=None,context=None):
        # with temporal blocks
        bhw = x.shape[0]
        batch_size = bhw // self.height // self.width
        x = rearrange(x,'(b h w) n c -> (b n) (h w) c',b=batch_size,h=self.height,w=self.width,n=self.movie_len).contiguous()
        # x = rearrange(x,'b c h w -> b (h w) c').contiguous()
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

class CrossAttention_Video(nn.Module):
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            backend=None,
            zero_init=False,
            **kwargs):
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
            nn.Dropout(dropout)
        )
        self.backend = backend

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            n_times_crossframe_attn_in_self=0,
            **kwargs
    ):
        num_heads = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            #add additional token
            x = torch.cat((additional_tokens,x),dim=1)

        q = self.to_q(x)
        context = default(context,x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],"b ... -> (b n) ...",n=n_cp,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],"b ... -> (b n) ...",n=n_cp,
            )
        q,k,v = map(lambda t:rearrange(t,"b n (h d) -> b h n d",h=num_heads).contiguous(),(q,k,v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            out = F.scaled_dot_product_attention(q,k,v,attn_mask=mask) # scale is dim_head ** -0.5 per default
        del q,k,v
        out = rearrange(out,"b h n d -> b n (h d)",h=num_heads).contiguous()

        if additional_tokens is not None:
            # remove additional_token 
            out = out[:,n_tokens_to_mask:]
        return self.to_out(out)

class MemoryEfficientCrossAttention(nn.Module):# we are using this implementation
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            zero_init=False,
            causual=False,
            add_lora=False,
            lora_rank=16,
            lora_scale=1.0,
            action_control=False,
            **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. "
            f"Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a dimension of {dim_head}"
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim,query_dim)
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k = nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v = nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,query_dim),
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None
        if causual:
            self.attn_bias = xformers.ops.LowerTriangularMask()
        else:
            self.attn_bias = None

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)
        
        self.add_lora = add_lora
        if add_lora:
            self.lora_scale = lora_scale

            self.q_adapter_down = nn.Linear(query_dim, lora_rank, bias=False)
            nn.init.normal_(self.q_adapter_down.weight, std=1 / lora_rank)
            self.q_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.q_adapter_up.weight)

            self.k_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.k_adapter_down.weight, std=1 / lora_rank)
            self.k_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_up.weight)

            self.v_adapter_down = nn.Linear(context_dim, lora_rank, bias=False)
            nn.init.normal_(self.v_adapter_down.weight, std=1 / lora_rank)
            self.v_adapter_up = nn.Linear(lora_rank, inner_dim, bias=False)
            nn.init.zeros_(self.v_adapter_up.weight)

            self.out_adapter_down = nn.Linear(inner_dim, lora_rank, bias=False)
            nn.init.normal_(self.out_adapter_down.weight, std=1 / lora_rank)
            self.out_adapter_up = nn.Linear(lora_rank, query_dim, bias=False)
            nn.init.zeros_(self.out_adapter_up.weight)

        self.action_control = action_control
        if action_control:
            self.context_dim = context_dim
            self.k_adapter_action_control = nn.Linear(128*19,inner_dim,bias=False)
            nn.init.zeros_(self.k_adapter_action_control.weight)
            self.v_adapter_action_control = nn.Linear(128*19,inner_dim,bias=False)
            nn.init.zeros_(self.v_adapter_action_control.weight)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            n_times_crossframe_attn_in_self=0,
            batchify_xformers=False):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat((additional_tokens, x), dim=1)
        context = default(context,x)
        if self.action_control:
            context,context_ = context[:,:,:self.context_dim],context[:,:,self.context_dim:]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.add_lora:
            q += self.q_adapter_up(self.q_adapter_down(x)) * self.lora_scale
            k += self.k_adapter_up(self.k_adapter_down(context)) * self.lora_scale
            v += self.v_adapter_up(self.v_adapter_down(context)) * self.lora_scale
        if self.action_control:
            k += self.k_adapter_action_control(context_)
            v += self.v_adapter_action_control(context_)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self
            )
        b,_,_ = q.shape
        q,k,v = map(
            lambda t: t.unsqueeze(3).
            reshape(b,t.shape[1],self.heads,self.dim_head)
            .permute(0,2,1,3)
            .reshape(b*self.heads,t.shape[1],self.dim_head)
            .contiguous(),
            (q,k,v)
        )
        # print(f"q.shape:{q.shape},k.shape:{k.shape},v.shape:{v.shape}")
        if exists(mask):
            raise NotImplementedError
        else:
            # actually compute the attention, what we cannot get enough of
            if batchify_xformers:
                max_bs = 32768
                n_batches = math.ceil(q.shape[0] / max_bs)
                out = list()
                for i_batch in range(n_batches):
                    batch = slice(i_batch * max_bs,(i_batch + 1) * max_bs)
                    out.append(
                        
                        xformers.ops.memory_efficient_attention(
                            q[batch],
                            k[batch],
                            v[batch],
                            attn_bias=self.attn_bias,
                            op=self.attention_op
                        )
                    )
                out = torch.cat(out,0)
            else:
                out = xformers.ops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=self.attn_bias,
                    op=self.attention_op
                )
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        if self.add_lora:
            return self.to_out(out) + self.out_adapter_up(self.out_adapter_down(out)) * self.lora_scale
        else:
            return self.to_out(out)



class TimeMixSequential(nn.Sequential):
    def forward(self,x,context=None,timesteps=None):
        for layer in self:
            x = layer(x,context,timesteps)

class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention_Video,
        "softmax_xformers": MemoryEfficientCrossAttention
    }
    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.0,
                 context_dim=None,
                 gated_ff=True,
                 use_checkpoint=False,
                 timesteps=None,
                 ff_in=False,
                 inner_dim=None,
                 attn_mode="softmax",
                 disable_self_attn=False,
                 disable_temporal_crossattention=False,
                 switch_temporal_ca_to_sa=False,
                 add_lora=False,
                 action_control=False,):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim
        
        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim
        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim,dim_out=inner_dim,dropout=dropout,glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                add_lora=add_lora,
            ) # is a cross-attn
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                causal=False,
                add_lora=add_lora
            )
        self.ff = FeedForward(inner_dim,dim_out=dim,dropout=dropout,glu=gated_ff)

        if not disable_temporal_crossattention:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    casual=False,
                    add_lora=add_lora,
                ) # is a self-attn
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    add_lora=add_lora,
                    action_control=action_control,
                ) # is self-attn if context is None
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
        

    def forward(self,x,context=None,timesteps=None):
        if self.use_checkpoint:
            return checkpoint(self._forward,x,context,timesteps,use_reentrant=False)
        else:
            return self._forward(x,context,timesteps=timesteps)
        
    def _forward(self,x,context=None,timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or timesteps == self.timesteps
        timesteps = self.timesteps or timesteps
        B,S,C = x.shape
        x = rearrange(x,"(b t) s c -> (b s) t c",t=timesteps).contiguous()

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip
        if self.disable_self_attn:
            x = self.attn1(self.norm1(x),context=context,batchify_xformers=True) + x
        else:
            x = self.attn1(self.norm1(x),batchify_xformers=True) + x
        
        if hasattr(self,"attn2"):
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x),batchify_xformers=True) + x
            else:
                x = self.attn2(self.norm2(x),context=context,batchify_xformers=True) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x,"(b s) t c -> (b t) s c",s=S,b=B//timesteps,c=C,t=timesteps).contiguous()
        return x
    
    def get_last_layer(self):
        return self.ff.net[-1].weight

class BasicTransformerBlock_Video(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention_Video,
        "softmax_xformers": MemoryEfficientCrossAttention,
    }
    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            disable_self_attn=False,
            attn_mode="softmax",
            sdp_backend=None,
            add_lora=False,
            action_control=False,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode `{attn_mode}` is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. You are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive")
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. `pip install xformers==0.0.16`"
            else:
                print("Falling back to xformers efficient attention")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            context_dim=context_dim if self.disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora=add_lora) # is a self-attn if not self.disable_self_attn
        self.ff = FeedForward(dim,dropout=dropout,glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            add_lora = add_lora,
            action_control=action_control
        ) # is self-attn if context is None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
        
    def forward(self,x,context=None,additional_tokens=None,n_times_crossframe_attn_in_self=0):
        kwargs = {"x":x}

        if context is not None:
            kwargs.update({"context":context})
        if additional_tokens is not None:
            kwargs.update({"additional_tokens":additional_tokens})
        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self":n_times_crossframe_attn_in_self})

        if self.use_checkpoint:
            return checkpoint(self._forward,x,context,use_reentrant=False)
        else:
            return self._forward(**kwargs)
        
    def _forward(self,x,context=None,additional_tokens=None,n_times_crossframe_attn_in_self=0):
        # spatial self-attn
        x = self.attn1(self.norm1(x),context=context if self.disable_self_attn else None,
                       additional_tokens=additional_tokens,
                       n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0) + x
        # spatial cross-attn
        x = self.attn2(self.norm2(x),context=context,additional_tokens=additional_tokens) + x
        # feed forward
        x = self.ff(self.norm3(x)) + x
        return x
        

class SpatialTransformer_Video(nn.Module):

    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=0.0,
            context_dim=None,
            disable_self_attn=False,
            use_linear=False,
            attn_type="softmax",
            use_checkpoint=False,
            sdp_backend=None,
            add_lora=False,
            action_control=False):
        super().__init__()
        print(f"Constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim,(list,ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim,list):
            if depth != len(context_dim):
                print(
                    f"WARNING: "
                    f"{self.__class__.__name__}: found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified depth of {depth}. "
                    f"Setting context_dim to {depth * [context_dim[0]]} now"
                )
                # depth does not match context dims
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "Need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if use_linear:
            self.proj_in = nn.Linear(in_channels,inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels,inner_dim,kernel_size=1,stride=1,padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_Video(inner_dim,
                                      n_heads,
                                      d_head,
                                      dropout=dropout,
                                      context_dim=context_dim[d],
                                      disable_self_attn=disable_self_attn,
                                      attn_mode=attn_type,
                                      use_checkpoint=use_checkpoint,
                                      sdp_backend=sdp_backend,
                                      add_lora=add_lora,
                                      action_control=action_control)
                for d in range(depth)
            ]
        )
        if use_linear:
            self.proj_out = zero_module(
                nn.Linear(inner_dim,in_channels)
            )
        else:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,in_channels,kernel_size=1,stride=1,padding=0)
            )
        self.use_linear = use_linear

    def forward(self,x,context=None):
        if not isinstance(context,list):
            context = [context]
        b,c,h,w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x,'b c h w -> b (h w) c').contiugous()
        if self.use_linear:
            x = self.proj_in(x)
        for i,block in enumerate(self.transformer_blocks):
            if i>0 and len(context) == 1:
                i = 0
            x = block(x,context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x,'b (h w) c -> b c h w',h=h,w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialVideoTransformer(SpatialTransformer_Video):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str="fixed",
        merge_factor: float=0.5,
        time_context_dim=None,
        ff_in=False,
        use_checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period=10000,
        add_lora=False,
        action_control=False,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=use_checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            add_lora=add_lora,
            action_control=action_control
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head

        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    use_checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    inner_dim = time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    add_lora=add_lora,
                    action_control=action_control
                )
                for _ in range(self.depth)
            ]
        )
        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(in_channels,time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim,in_channels)
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern = "b t -> (b t) 1 1"
        )

    def forward(
            self,
            x,
            context=None,
            time_context=None,
            timesteps=None,
    ):
        _,_,h,w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context
        
        if self.use_spatial_context:
            assert context.ndim == 3
            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)
        elif time_context is not None and self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c").contiguous()
        
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x,'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps,device=x.device)
        num_frames = repeat(num_frames,"t -> (b t)",b=x.shape[0] // timesteps)
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:,None]

        for block,mix_block in zip(self.transformer_blocks,self.time_stack):
            x = block(x,context=spatial_context)

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix,context=time_context,timesteps=timesteps)
            x = self.time_mixer(x_spatial=x,x_temporal=x_mix)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x,"b (h w) c -> b c h w",h=h,w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out