import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention,TemporalAttention
from ldm.modules.distributions.distributions import DiagonalGuassianDistribution
from ldm.modules.diffusionmodules.openaimodel import VResBlock 
from typing import Iterable

def get_timestep_embedding(timesteps,embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim,dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:,None] * emb[None,:]
    emb = torch.cat([torch.sin(emb),torch.cos(emb)],dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb,(0,1,0,0))
    return emb

def nonlinearity(x):
    #swish 
    return x * torch.sigmoid(x)

def Normalize(in_channels,num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups,num_channels=in_channels,eps=1e-6,affine=True)

#TODO:scale 4 maybe need change
class Upsample(nn.Module):
    def __init__(self,in_channels,with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    def forward(self,x):
        x = torch.nn.functional.interpolate(x,scale_factor=4.0,mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self,in_channels,with_conv,out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels or in_channels
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=4,
                                        padding=0)
    def forward(self,x):
        if self.with_conv:
            # pad = (0,1,0,1)
            # x = torch.nn.functional.pad(x,pad,mode='constant',value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x,kernel_size=4,stride=4)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self,*,in_channels,out_channels=None,conv_shortcut=False,
                 dropout,temb_channels=512,num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels,num_groups)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,out_channels)
        
        self.norm2 = Normalize(out_channels,num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shorcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
    def forward(self,x,temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
            
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shorcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    

class VideoResnetBlock(ResnetBlock):
    def __init__(self,in_channels,out_channels=None,temb_channels=512,num_groups=32,dropout=0.,merge_strategy='learned',alpha=0.0,movie_len=None,video_kernel_size=[3,1,1],):
        super().__init__(in_channels=in_channels,out_channels=out_channels,temb_channels=temb_channels,num_groups=num_groups,dropout=dropout)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = VResBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=False,
            skip_t_emb=True
        )
        self.merge_strategy = merge_strategy
        self.movie_len = movie_len
        if self.merge_strategy == 'fixed':
            self.register_buffer("mix_factor",torch.Tensor([alpha]))
        elif self.merge_strategy == 'learned':
            self.register_buffer('mix_factor',torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError    
    
    def forward(self,x,time_emb,skip_video=False,timesteps=None):
        if timesteps is None:
            timesteps = self.movie_len
        x = super().forward(x,time_emb)
        if not skip_video:
            x_mix = rearrange(x,'(b n) c h w -> b c n h w',n=timesteps).contiguous()
            x = rearrange(x,'(b n) c h w -> b c n h w',n=timesteps).contiguous()
            x = self.time_stack(x,time_emb)
            alpha = self.get_alpha()
            x = alpha * x + (1.0 - alpha) * x_mix
            x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        return x
    
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self,in_channels):
        super().__init__(dim=in_channels,heads=1,dim_head=in_channels)

class AttnBlock(nn.Module):
    def __init__(self,in_channels,num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels,num_groups)
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
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)

        #attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

#TODO:add gate self attention and temporal attention

def make_attn(in_channels,num_groups=32,attn_type="vanilla"):
    assert attn_type in ['vanilla','linear','none'],f'attn_type {attn_type} unkown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == 'vanilla':
        return AttnBlock(in_channels,num_groups)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
    
class Model(nn.Module):
    def __init__(self,*,ch,out_ch,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,use_timestep=True,use_linear_attn=False,attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,self.temb_ch),
                torch.nn.Linear(self.temb_ch,self.temb_ch)
            ])

        #downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in,resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        
        #middle 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        #upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,attn_type=attn_type))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in,resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0,up)
        
        #end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self,x,t=None,context=None):
        pass

# class MyEncoder(nn.Module):
#     def __init__(self,in_channel,ch,dropout=0.0,ch_mult=(1,1,2)):
#         super().__init__()
#         self.in_channel = in_channel
#         self.ch_mult = ch_mult
#         self.dropout = dropout
#         self.ch = ch
#         self.conv_in = torch.nn.Conv2d(in_channel,ch,kernel_size=3,stride=1)
#         self.in_ch_mult = (1,) + tuple(ch_mult)
#         self.num_resolutions = len(ch_mult)
#         for i_level in range(0,self.num_resolutions):
#             block = nn.ModuleList()
#             block_in = ch * self.in_ch_mult[i_level]
#             block_out = ch * self.ch_mult[i_level]
            


class Encoder(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,double_z=True,use_linear_attn=False,attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        #downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,self.ch,kernel_size=3,stride=1,padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         num_groups=num_groups,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in,resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,num_groups, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self,x):
        #timestep embedding
        temb = None

        #downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1],temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        #middle
        h = hs[-1]
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        #end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class Decoder(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups=32,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,give_pre_end=False,tanh_out=False,use_linear_attn=False,
                 attn_type="vanilla",**ignorekwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        #compute in_ch_mult,block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape,np.prod(self.z_shape)))
        
        #z to block in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        #middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,num_groups,attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        
        #upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         num_groups=num_groups,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in,resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0,up) #prepend to get consistent order

        #end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self,z):
        self.last_z_shape = z.shape

        #timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        #middle
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        #upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h,temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        #end
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    

class SimpleDecoder(nn.Module):
    def __init__(self,in_channels,out_channels,*args,**kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels,in_channels,1),
                                    ResnetBlock(in_channels=in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0,dropout=0.0),
                                    ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0,dropout=0.0),
                                    ResnetBlock(in_channels=4*in_channels,
                                                out_channels=2*in_channels,
                                                temb_channels=0,dropout=0.0),
                                    nn.Conv2d(2*in_channels,in_channels,1),
                                    Upsample(in_channels,with_conv=True)])
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        for i,layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x,None)
            else:
                x = layer(x)
        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x
    
class UpsampleDecoder(nn.Module):
    def __init__(self,in_channels,out_channels,ch,num_res_blocks,resolution,
                 ch_mult=(2,2),dropout=0.0):
        super().__init__()
        #upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in,True))
                curr_res = curr_res * 2
            
        #end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,out_channels,
                                        kernel_size=3,stride=1,padding=1)
        
    def forward(self,x):
        #upsampling
        h = x
        for k,i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h,None)
            
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
# in_channels -> mid_channels ->mid_channels ->mid_channels->out_channels
# image_size -> image_size * scaler
class LatentRescaler(nn.Module):
    def __init__(self,factor,in_channels,mid_channels,out_channels,depth=2):
        super().__init__()
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1)
        
    def forward(self,x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x,None)
        x = torch.nn.functional.interpolate(x,size=(int(round(x.shape[2]*self.factor)),int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x= block(x,None)
        x = self.conv_out(x)
        return x
    
class MergedRescalerEncoder(nn.Module):
    def __init__(self,in_channels,ch,resolution,out_ch,num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,
                 ch_mult=(1,2,4,8),rescale_factor=1.0,rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels,num_res_blocks=num_res_blocks,ch=ch,ch_mult=ch_mult,
                               z_channels=intermediate_chn,double_z=False,resolution=resolution,
                               attn_resolutions=attn_resolutions,dropout=dropout,resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor,in_channels=intermediate_chn,mid_channels=intermediate_chn,out_channels=out_ch,depth=rescale_module_depth)

    def forward(self,x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x
    
class Upsampler(nn.Module):
    def __init__(self,in_size,out_size,in_channels,out_channels,ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1. + (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up,in_channels=in_channels,mid_channels=2*in_channels,
                                      out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels,resolution=out_size,z_channels=in_channels,num_res_blocks=2,
                               attn_resolutions=[],in_channels=None,ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])
        
    def forward(self,x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x
    
class Resize(nn.Module):
    def __init__(self,in_channels=None,learned=False,mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
    def forward(self,x,scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x,mode=self.mode,align_corners=False,scale_factor=scale_factor)
        return x
    

class FirstStagePostProcessor(nn.Module):
    def __init__(self,ch_mult:list,in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)
        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in,with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)

    def instantiate_pretrained(self,config):
        model = instantiate_from_config(config=config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c,DiagonalGuassianDistribution):
            c = c.mode()
        return c
    
    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)
        for submodel,downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)
        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z
    
class Encoder_Diffusion(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,double_z=True,use_linear_attn=False,attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        #downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,self.ch,kernel_size=3,stride=1,padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         num_groups=num_groups,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample_Diffusion(block_in,resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,num_groups, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self,x):
        #timestep embedding
        temb = None

        #downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1],temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        #middle
        h = hs[-1]
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        #end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder_Diffusion(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups=32,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,give_pre_end=False,tanh_out=False,use_linear_attn=False,
                 attn_type="vanilla",**ignorekwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        #compute in_ch_mult,block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape,np.prod(self.z_shape)))
        
        #z to block in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        #middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,num_groups,attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        
        #upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         num_groups=num_groups,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample_Diffusion(block_in,resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0,up) #prepend to get consistent order

        #end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self,z,return_temp_output=False):
        self.last_z_shape = z.shape

        #timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        #middle
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        if return_temp_output:
            temp = []
        #upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h,temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if return_temp_output:
                    temp.append(h)

        #end
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        if return_temp_output:
            return h,temp
        return h

class Encoder_Temporal(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,double_z=True,use_linear_attn=False,attn_type="vanilla",movie_len=None,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.movie_len = movie_len
        #downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,self.ch,kernel_size=3,stride=1,padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(VideoResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         num_groups=num_groups,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         movie_len=self.movie_len))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample_Diffusion(block_in,resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VideoResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       movie_len=self.movie_len)
        self.mid.attn_1 = make_attn(block_in,num_groups, attn_type=attn_type)
        self.mid.block_2 = VideoResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       movie_len=self.movie_len)

        # end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self,x):
        #timestep embedding
        temb = None

        #downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1],temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        #middle
        h = hs[-1]
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        #end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class AE3DConv(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, movie_len=None,*args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, Iterable):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)
        self.movie_len = movie_len
        self.time_mix_conv = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding
        )

    def forward(self, input, skip_video=False):
        x = super().forward(input)
        if skip_video:
            return x
        else:
            x = rearrange(x, "(b t) c h w -> b c t h w", t=self.movie_len)
            x = self.time_mix_conv(x)
            return rearrange(x, "b c t h w -> (b t) c h w")

class Decoder_Temporal(nn.Module):
    def __init__(self,*,ch,out_ch,num_groups=32,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,movie_len=None,give_pre_end=False,tanh_out=False,use_linear_attn=False,
                 attn_type="vanilla",**ignorekwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.movie_len = movie_len

        #compute in_ch_mult,block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape,np.prod(self.z_shape)))
        
        #z to block in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        #middle
        self.mid = nn.Module()
        self.mid.block_1 = VideoResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout,
                                       movie_len=self.movie_len)
        self.mid.attn_1 = make_attn(block_in,num_groups,attn_type=attn_type)
        self.mid.block_2 = VideoResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout,
                                       movie_len=self.movie_len)
        
        #upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(VideoResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         num_groups=num_groups,
                                         dropout=dropout,
                                         movie_len=self.movie_len))
                # if not self.movie_len is None:
                #     block.append(TemporalAttention(channels=block_out,num_heads=8,num_head_channels=64,movie_len=self.movie_len))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample_Diffusion(block_in,resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0,up) #prepend to get consistent order

        #end
        self.norm_out = Normalize(block_in,num_groups)
        self.conv_out = AE3DConv(block_in,
                                out_ch,
                                kernel_size=3,
                                video_kernel_size=[3,1,1],
                                stride=1,
                                padding=1,
                                movie_len = self.movie_len)
        
    def forward(self,z,return_temp_output=False):
        self.last_z_shape = z.shape

        #timestep embedding
        temb = None
        # z to block_in
        h = self.conv_in(z)

        #middle
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)
        if return_temp_output:
            temp = []
        #upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                if isinstance(self.up[i_level].block[i_block],TemporalAttention):
                    b,c,height,width = h.shape
                    b = b // self.movie_len
                    n = self.movie_len
                    h = self.up[i_level].block[i_block](h)
                    h = rearrange(h,'(b h w) n c -> (b n) c h w',b=b,n=n,c=c,h=height,w=width)
                else:
                    h = self.up[i_level].block[i_block](h,temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if return_temp_output:
                    temp.append(h)

        #end
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        if return_temp_output:
            return h,temp
        return h
    

class Upsample_Diffusion(nn.Module):
    def __init__(self,in_channels,with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    def forward(self,x):
        x = torch.nn.functional.interpolate(x,scale_factor=2.0,mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
    
class Downsample_Diffusion(nn.Module):
    def __init__(self,in_channels,with_conv,out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels or in_channels
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self,x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x,pad,mode='constant',value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x,kernel_size=4,stride=4)
        return x
    

class Decoder_Lidar(nn.Module):
    def __init__(self,*,ch,label_out_ch,img_out_ch,num_groups=32,ch_mult=(1,2,4,8),num_res_blocks,
                 attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,
                 resolution,z_channels,give_pre_end=False,tanh_out=False,use_linear_attn=False,
                 attn_type="vanilla",**ignorekwargs):
        super().__init__()
        if use_linear_attn:attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        #compute in_ch_mult,block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape,np.prod(self.z_shape)))
        
        #z to block in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        #middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,num_groups,attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       num_groups=num_groups,
                                       dropout=dropout)
        
        #upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         num_groups=num_groups,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in,num_groups,attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample_Diffusion(block_in,resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0,up) #prepend to get consistent order

        #end
        self.norm_out = Normalize(block_in,num_groups)
        self.label_out = torch.nn.Conv2d(block_in,
                                        label_out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.img_out = torch.nn.Conv2d(block_in,
                                       img_out_ch,
                                       kernel_size=3,
                                        stride=1,
                                        padding=1) 
        
    def forward(self,z):
        self.last_z_shape = z.shape

        #timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        #middle
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)

        #upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h,temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        #end
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        label = self.label_out(h)
        img = self.img_out(h)
        h = torch.cat([label,img],dim=1)
        if self.tanh_out:
            h = torch.tanh(h)
        return h