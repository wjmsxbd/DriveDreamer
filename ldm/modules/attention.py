from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


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


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,use_image_clip=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.use_image_clip = use_image_clip
        if use_image_clip:
            self.attn3 = CrossAttention(query_dim=dim,context_dim=context_dim,
                                        heads=n_heads,dim_head=d_head,dropout=dropout) # is cross-attn if context is none
            self.norm4 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        if not self.use_image_clip:
            x = self.attn2(self.norm2(x), context=context) + x
        else:
            x = self.attn2(self.norm2(x), context=context[:,1:]) + x
            x = self.attn3(self.norm4(x),context=context[:,0:1]) + x
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
                 depth=1, dropout=0., context_dim=None,use_image_clip=False):
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
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,use_image_clip=use_image_clip)
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
    def __init__(self,input_channels,attention_head_dim,norm_num_groups=32):
        super().__init__()
        self.attn1 = CrossAttention(input_channels,input_channels,heads=input_channels//attention_head_dim,dim_head=attention_head_dim)
        self.norm1 = torch.nn.GroupNorm(
            num_groups=norm_num_groups,num_channels=input_channels,eps=1e-6,affine=True
        )
        # self.attn2 = CrossAttention(input_channels,input_channels,heads=input_channels//attention_head_dim,dim_head=attention_head_dim)
        # self.norm2 = torch.nn.GroupNorm(
        #     num_groups=norm_num_groups,num_channels=input_channels,eps=1e-6,affine=True
        # )
        self.proj_in = nn.Linear(input_channels,input_channels)
        self.proj_out = nn.Linear(input_channels,input_channels)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self,hidden_state,encoder_hidden_states):
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
        hidden_state_norm = self.norm1(hidden_state)
        hidden_state_norm = rearrange(hidden_state_norm,'b c h w -> (b h w) c',h=h,w=w)
        hidden_state_norm = hidden_state_norm.unsqueeze(1)
        hidden_state_norm = self.proj_in(hidden_state_norm)
        encoder_hidden_states = rearrange(encoder_hidden_states,'b n c h w -> (b h w) n c')
        attn = self.attn1(hidden_state_norm,encoder_hidden_states,mask=None)
        attn = rearrange(attn.squeeze(1),'(b hw) c -> b hw c',hw=h*w)
        # encoder_hidden_states = rearrange(encoder_hidden_states,'(b hw) n c -> (b n) hw c',hw=h*w)

        # attn = rearrange(attn.squeeze(1),'(b h w) c -> b c h w ',h=h,w=w)
        # attn = self.norm2(attn)
        # attn = rearrange(attn,'b c h w -> b (h w) c')
        # print(attn.shape)
        # print(encoder_hidden_states.shape)
        # attn2 = self.attn2(attn,encoder_hidden_states) + attn

        residual = self.proj_out(attn)

        hidden_state = self.dropout(hidden_state)

        residual = rearrange(residual,'b (h w) c -> b c h w',h=h,w=w)
        output = residual + hidden_state
        return output
