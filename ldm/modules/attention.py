from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import copy
# from ldm.modules.diffusionmodules.util import checkpoint
from typing import Union,List,Optional,Any
from torch.utils.checkpoint import checkpoint
from packaging import version
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3).contiguous()
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w).contiguous()
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
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

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()
        k = rearrange(k, 'b c h w -> b c (h w)').contiguous()
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)').contiguous()
        w_ = rearrange(w_, 'b i j -> b j i').contiguous()
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h).contiguous()
        h_ = self.proj_out(h_)

        return x+h_


class RopeCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.,seq_len=256,choose_feature_idx=[-10,-5,-1]):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim,query_dim)

        self.scale = dim_head ** 0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.positional_encoding = RotaryPositionalEmbeddings(d=dim_head,seq_len=seq_len)
        self.choose_feature_idx = choose_feature_idx

    def get_positional_encoding(self,x,frames,additional_frame=False):
        if additional_frame:
            batch_indices = []
            for i in range(len(frames)):
                frame = frames[i]
                indices = []
                for idx in self.choose_feature_idx:
                    if frame + idx < 0:
                        indices.append(0)
                    else:
                        indices.append(frame + idx)
                batch_indices.append(indices)
            x = self.positional_encoding(x,batch_indices)
        else:
            x = self.positional_encoding(x,frames)
        return x

    def forward(self,x,frames,context=None,mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context,x)
        k = self.to_k(context)
        v = self.to_v(context)

        q,k = map(lambda t: rearrange(t,'b n (h d) -> b h n d',h=h).contiguous(),(q,k))
        q = self.get_positional_encoding(q,frames)
        k = self.get_positional_encoding(k,frames,additional_frame=True)
        q,k = map(lambda t: rearrange(t,'b h n d -> (b h) n d').contiguous(),(q,k))
        v = rearrange(v,'b n (h d) -> (b h) n d', h=h).contiguous()
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
        

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h).contiguous(), (q, k, v))

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

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)

class RopeMemoryEfficientCrossAttention(MemoryEfficientCrossAttention):
    def __init__(self,query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,seq_len=256,choose_feature_idx=[-10,-5,-1],**kwargs):
        super().__init__(query_dim,context_dim,heads,dim_head,dropout,**kwargs)
        self.positional_encoding = RotaryPositionalEmbeddings(d=dim_head,seq_len=seq_len)
        self.choose_feature_idx = choose_feature_idx

    def get_positional_encoding(self,x,frames,additional_frame=False):
        if additional_frame:
            batch_indices = []
            for i in range(len(frames)):
                frame = frames[i]
                indices = []
                for idx in self.choose_feature_idx:
                    if frame + idx < 0:
                        indices.append(0)
                    else:
                        indices.append(frame + idx)
                batch_indices.append(indices)
            x = self.positional_encoding(x,batch_indices)
        else:
            x = self.positional_encoding(x,frames)
        return x
    
    def forward(self,x,frames,context=None,mask=None):
        b = x.shape[0]
        h = self.heads
        q = self.to_q(x)
        context = default(context,x)
        k = self.to_k(context)
        v = self.to_v(context)

        q,k = map(lambda t: rearrange(t,'b n (h d) -> b h n d',h=h).contiguous(),(q,k))
        q = self.get_positional_encoding(q,frames)
        k = self.get_positional_encoding(k,frames,additional_frame=True)
        q,k = map(lambda t: rearrange(t,'b h n d -> (b h) n d').contiguous(),(q,k))
        v = rearrange(v,'b n (h d) -> (b h) n d', h=h).contiguous()

        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict



class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, num_cameras=1,gated_ff=True, checkpoint=True,use_image_clip=False,attn_type='softmax'):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[attn_type]
        neighboring_view_pair = {
                        0: [5, 1],
                        1: [0, 2],
                        2: [1, 3],
                        3: [2, 4],
                        4: [3, 5],
                        5: [4, 0]
                                    }
        self.num_cameras = num_cameras
        
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.use_image_clip = use_image_clip
        if use_image_clip:
            self.attn3 = attn_cls(query_dim=dim,context_dim=context_dim,
                                        heads=n_heads,dim_head=d_head,dropout=dropout) # is cross-attn if context is none
            self.norm4 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type =  "add"
        # multiview attention
        self.norm5 = nn.LayerNorm(dim)
        self.attn5 = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.connector = zero_module(nn.Linear(dim, dim))
        self.checkpoint = checkpoint

    @property
    def n_cam(self):
        return self.num_cameras

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order


    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        return checkpoint(self._forward,x,context,use_reentrant=False)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        if not self.use_image_clip:
            x = self.attn2(self.norm2(x), context=context) + x
        else:
            x = self.attn2(self.norm2(x), context=context[:,1:]) + x
            x = self.attn3(self.norm4(x),context=context[:,0:1]) + x
        # multi-view cross attention
        if self.n_cam > 1 :
            norm_hidden_states = (
                self.norm5(x)
            )
            # batch dim first, cam dim second
            norm_hidden_states = rearrange(
                norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
            B = len(norm_hidden_states)
            # key is query in attention; value is key-value in attention
            hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
                norm_hidden_states, )
            # attention
            attn_raw_output = self.attn5(
                hidden_states_in1,
                context=hidden_states_in2,
            )
            # final output
            if self.neighboring_attn_type == "self":
                attn_output = rearrange(
                    attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
            else:
                attn_output = torch.zeros_like(norm_hidden_states)
                for cam_i in range(self.n_cam):
                    attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                            '(n b) ... -> b n ...', b=B)
                    attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
            # apply zero init connector (one layer)
            attn_output = self.connector(attn_output)
            # short-cut
            x = attn_output + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,num_cameras=1,use_image_clip=False,attn_type='softmax'):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,num_cameras = num_cameras,use_image_clip=use_image_clip,attn_type=attn_type)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in
    
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self,d:int,seq_len:int=256,base:int=10000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None
        self.seq_len = seq_len
        self._build_cache()

    def _build_cache(self):
        
        theta = 1. / (self.base ** (torch.arange(0,self.d,2).float() / self.d))
        seq_idx = torch.arange(self.seq_len).float()
        idx_theta = torch.einsum('n,d->nd',seq_idx,theta)
        idx_theta2 = torch.cat([idx_theta,idx_theta],dim=1) # seq d
        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    def _neg_half(self,x:torch.Tensor):
        return torch.cat([-x[:,:,:,:,1::2],x[:,:,:,:,::2]],dim=-1)
    
    # return b hw h n d
    def get_cosine_cache(self,indices:List):
        # assert type(indices[0]) == list
        cache = []
        print(indices)
        for idx in indices:
            cache.append(self.cos_cached[idx])
        cache = torch.stack(cache,dim=0)
        while len(cache.shape) < 5:
            cache = cache[:,None]
        return copy.deepcopy(cache)

    def get_sine_cache(self,indices:List):
        # assert type(indices[0]) == list
        cache = []
        for idx in indices:
            cache.append(self.sin_cached[idx])
        cache = torch.stack(cache,dim=0)
        while len(cache.shape) < 5:
            cache = cache[:,None]
        return copy.deepcopy(cache)

    # x:[bs,head,n,d]
    def forward(self,x:torch.Tensor,indices:List):
        hw = x.shape[0] // len(indices)
        x = rearrange(x,'(b hw) h n c -> b hw h n c',hw=hw)
        neg_half_x = self._neg_half(x)
        x = x * self.get_cosine_cache(indices).to(x.device) + neg_half_x * self.get_sine_cache(indices).to(x.device)
        x = rearrange(x,'b hw h n c -> (b hw) h n c')
        return x
        

class PixelTemporalAttention(nn.Module):
    """
    CrossAttention module implements per-pixel temporal attention to fuse the conditional attention module with the base module.

    Args:
        input_channels (int): Number of input channels.
        attention_head_dim (int): Dimension of attention head.
        norm_num_groups (int): Number of groups for GroupNorm normalization (default is 32).

    Attributes:
        attention (Attention): Attention module for computing attention scores.
        norm (torch.nn.GroupNorm): Group normalization layer.
        proj_in (nn.Linear): Linear layer for projecting input data.
        proj_out (nn.Linear): Linear layer for projecting output data.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(hidden_state, encoder_hidden_states, num_frames, num_conditional_frames):
            Forward pass of the CrossAttention module.

    """
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }
    def __init__(self,input_channels,attention_head_dim,norm_num_groups=32,use_rope_positional_encoding=False,seq_len=256,choose_feature_idx=[-10,-5,-1],attn_type="softmax"):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[attn_type]
        if use_rope_positional_encoding:
            self.attn1 = attn_cls(input_channels,input_channels,heads=input_channels//attention_head_dim,dim_head=attention_head_dim,seq_len=seq_len,choose_feature_idx=choose_feature_idx)
        else:
            self.attn1 = attn_cls(input_channels,input_channels,heads=input_channels//attention_head_dim,dim_head=attention_head_dim)
        self.norm1 = torch.nn.GroupNorm(
            num_groups=norm_num_groups,num_channels=input_channels,eps=1e-6,affine=True
        )
        self.attn2 = attn_cls(input_channels,input_channels,heads=input_channels//attention_head_dim,dim_head=attention_head_dim)
        self.norm2 = nn.LayerNorm(input_channels)
        self.proj_in = nn.Linear(input_channels,input_channels)
        self.proj_out = nn.Linear(input_channels,input_channels)
        self.dropout = nn.Dropout(p=0.25)

    def _forward(self,hidden_state,encoder_hidden_states,frames,):
        """
        The input hidden state is normalized, then projected using a linear layer.
        Multi-head cross attention is computed between the hidden state (latent of noisy video) and encoder hidden states (CLIP image encoder).
        The output is projected using a linear layer.
        We apply dropout to the newly generated frames (without the control frames).

        Args:
            hidden_state (torch.Tensor): Input hidden state tensor.
            encoder_hidden_states (torch.Tensor): Encoder hidden states tensor.

        Returns:
            output (torch.Tensor): Output tensor after processing with attention mechanism.

        """
        h,w = hidden_state.shape[2],hidden_state.shape[3]
        b = hidden_state.shape[0]
        hidden_state_norm = self.norm1(hidden_state)
        hidden_state_norm = rearrange(hidden_state_norm,'b c h w -> (b h w) c',h=h,w=w)
        hidden_state_norm = hidden_state_norm.unsqueeze(1)
        hidden_state_norm = self.proj_in(hidden_state_norm)
        encoder_hidden_states = rearrange(encoder_hidden_states,'b n c h w -> (b h w) n c')
        if isinstance(self.attn1,RopeCrossAttention) or isinstance(self.attn1,RopeMemoryEfficientCrossAttention):
            attn = self.attn1(hidden_state_norm,frames,encoder_hidden_states,mask=None)    
        else:
            attn = self.attn1(hidden_state_norm,encoder_hidden_states,mask=None)
        attn = rearrange(attn.squeeze(1),'(b hw) c -> b hw c',hw=h*w)
        encoder_hidden_states = rearrange(encoder_hidden_states,'(b hw) n c -> (b n) hw c',hw=h*w)
        # attn = rearrange(attn,"(b hw) n c -> b n hw c",hw=h*w).contiguous()

        # attn = repeat(attn,'b n hw c -> b (repeat n) hw c',repeat=encoder_hidden_states.shape[0] // attn.shape[0])
        # attn = rearrange(attn,'b n hw c -> (b n) hw c').contiguous()
        # attn = self.norm2(attn)
        # attn2 = self.attn2(attn,encoder_hidden_states)
        # attn = rearrange(attn,'(b n) hw c -> b n hw c',b=b)
        # attn2 = rearrange(attn2,'(b n) hw c -> b n hw c',b=b)
        
        # attn = torch.sum(attn2,dim=1) + attn[:,0]
        
        

        residual = self.proj_out(attn)

        hidden_state = self.dropout(hidden_state)

        residual = rearrange(residual,'b (h w) c -> b c h w',h=h,w=w)
        output = residual + hidden_state
        return output
    
    def forward(self,hidden_state,encoder_hidden_states,frames):
        # return checkpoint(self._forward, (hidden_state,encoder_hidden_states,frames), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, hidden_state,encoder_hidden_states,frames,use_reentrant=False)
