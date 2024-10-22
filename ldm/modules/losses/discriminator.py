import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
sys.path.append('....')
import functools
import torch.nn as nn
import torch
from ldm.modules.attention import CrossAttention
from taming.modules.util import ActNorm
from einops import rearrange
from ldm.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
import pytorch_lightning as pl
import omegaconf
import torch.nn.functional as F
class CRLayerDiscriminator(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 4
        padw = 1
        attn = [CrossAttention(in_channels,context_dim=context_dim,heads=n_heads,dim_head=dim_head)]
        norm = [nn.LayerNorm(in_channels)]
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=kw,stride=2,padding=padw),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            attn.append(CrossAttention(ndf*nf_mult_prev,context_dim=context_dim,heads=n_heads,dim_head=dim_head))
            norm.append(nn.LayerNorm(ndf * nf_mult_prev))
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        norm.append(nn.LayerNorm( ndf * nf_mult_prev))
        attn.append(CrossAttention(ndf*nf_mult_prev,context_dim=context_dim,heads=n_heads,dim_head=dim_head))
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        self.attn = nn.ModuleList(attn)
        self.norm = nn.ModuleList(norm)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            # if idx != len(self.conv) - 1:
            #     input = rearrange(input,'b c h w -> b (h w) c').contiguous()
            #     input = self.attn[idx](self.norm[idx](input),cond) + input
            #     input = rearrange(input,'b (h w) c -> b c h w',h=h,w=w).contiguous()
            input = self.conv[idx](input)
            if idx<self.n_layers:
                h = h // 2
                w = w // 2
        return input

class CRLayerDiscriminator2(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator2,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 3
        padw = 1
        attn = [CrossAttention(in_channels,context_dim=context_dim,heads=n_heads,dim_head=dim_head)]
        norm = [nn.LayerNorm(in_channels)]
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=kw,stride=2,padding=padw),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            attn.append(CrossAttention(ndf*nf_mult_prev,context_dim=context_dim,heads=n_heads,dim_head=dim_head))
            norm.append(nn.LayerNorm(ndf * nf_mult_prev))
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        norm.append(nn.LayerNorm( ndf * nf_mult_prev))
        attn.append(CrossAttention(ndf*nf_mult_prev,context_dim=context_dim,heads=n_heads,dim_head=dim_head))
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        self.attn = nn.ModuleList(attn)
        self.norm = nn.ModuleList(norm)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
        return input

class CRLayerDiscriminator2(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator2,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 3
        padw = 1
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=kw,stride=2,padding=padw),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
        return input
    
class CRLayerDiscriminator3(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator3,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 3
        padw = 1
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
        return input

class CRLayerDiscriminator4(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator4,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 3
        padw = 1
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=kw,stride=1,padding=padw),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
        return input
    
class CRLayerDiscriminator_Video(nn.Module):
    def __init__(self,in_channels,movie_len,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator_Video,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        self.movie_len = movie_len
        kw = 3
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        padw = 1
        conv3d = [nn.Conv3d(ndf,ndf,kernel_size=(3,3,3),stride=1,padding=(1,1,1))]
        sequence = [nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=kw,stride=1,padding=padw),nn.LeakyReLU(0.2,True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            conv3d.append(nn.Conv3d(ndf*nf_mult,ndf*nf_mult,kernel_size=3,stride=1,padding=1))
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv3d.append(nn.Conv3d(ndf*nf_mult,ndf*nf_mult,kernel_size=3,stride=1,padding=1))
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv = nn.ModuleList(sequence)
        self.conv3d = nn.ModuleList(conv3d)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
            if idx != len(self.conv) - 1:
                input = rearrange(input,'(b n) c h w -> b c n h w',n=self.movie_len).contiguous()
                input = self.conv3d[idx](input)
                input = rearrange(input,'b c n h w -> (b n) c h w').contiguous()
        return input

class CRLayerDiscriminator_Pipe(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=3,use_actnorm=False,context_dim=1024,n_heads=8,dim_head=64,ckpt_path=None,ignore_keys=[],schedule_config=None):
        super(CRLayerDiscriminator_Pipe,self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        if not schedule_config is None:
            self.scheduler = instantiate_from_config(schedule_config)
        kw = 4
        padw = 1
        sequence_1 = [
            nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=3,stride=1,padding=padw),norm_layer(ndf),nn.LeakyReLU(0.2,True)),
            nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=padw),norm_layer(ndf),nn.LeakyReLU(0.2,True)),
        ]
        sequence_2 = [
            nn.Sequential(nn.Conv2d(in_channels,ndf,kernel_size=3,stride=1,padding=1),norm_layer(ndf),nn.LeakyReLU(0.2,True)),
            nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=padw),norm_layer(ndf),nn.LeakyReLU(0.2,True)),
        ]
        sequence = []
        nf_mult = 2
        nf_mult_prev = 2
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            sequence.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        sequence.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)))  # output 1 channel prediction map
        self.conv_1 = nn.ModuleList(sequence_1)
        self.conv_2 = nn.ModuleList(sequence_2)
        self.conv = nn.ModuleList(sequence)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        new_dict = dict()
        for name,value in sd.items():
            if name.startswith('discriminator'):
                new_dict[name[len('discriminator.'):]] = value
        # keys = list(sd.keys())
        # for k in keys:
        #     for ik in ignore_keys:
        #         if k.startswith(ik):
        #             print("Deleting key {} from state_dict".format(k))
        #             del sd[k]
        missing,unexpected = self.load_state_dict(new_dict,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self,input,cond=None):
        b,c,h,w = input.shape
        x,range_x = torch.chunk(input,2,1)
        for idx in range(len(self.conv_1)):
            x = self.conv_1[idx](x)
            range_x = self.conv_2[idx](range_x)
        input = torch.cat([x,range_x],dim=1)
        for idx in range(len(self.conv)):
            input = self.conv[idx](input)
        return input

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def new_d_loss(logits_real,logits_fake):
    logits_real = torch.mean(F.relu(1. - logits_real))
    logits_fake_1,logits_fake_2,logits_fake_3 = torch.chunk(logits_fake,3,0)
    logits_fake_1 = torch.mean(F.relu(1. + logits_fake_1))
    logits_fake_2 = torch.mean(F.relu(1. + logits_fake_2))
    logits_fake_3 = torch.mean(F.relu(1. + logits_fake_3))
    logits_fake = logits_fake_1 + 0.5 * logits_fake_2 + 0.5 * logits_fake_3
    d_loss = 2 * logits_real + logits_fake
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def vanilla_new_d_loss(logits_real,logits_fake):
    logits_real = torch.mean(torch.nn.functional.softplus(-logits_real))
    logits_fake_1,logits_fake_2,logits_fake_3 = torch.chunk(logits_fake,3,0)
    logits_fake_1 = torch.mean(torch.nn.functional.softplus(logits_fake_1))
    logits_fake_2 = torch.mean(torch.nn.functional.softplus(logits_fake_2))
    logits_fake_3 = torch.mean(torch.nn.functional.softplus(logits_fake_3))
    logits_fake = logits_fake_1 + 0.5 * logits_fake_2 + 0.5 * logits_fake_3
    d_loss = (2 * logits_real + logits_fake) / 4
    return d_loss



class PipeLineCRLayerDiscriminator(pl.LightningModule):
    def __init__(self,discriminator_config,global_condition_config,cond_stage_config,movie_len,scheduler_config=None,monitor='val/loss',disc_factor=1.0,disc_weight=1.0,disc_loss='hinge',disc_start=0,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.discriminator = instantiate_from_config(discriminator_config)
        self.global_condition = instantiate_from_config(global_condition_config)
        self.clip = instantiate_from_config(cond_stage_config)
        self.clip = self.clip.eval()
        self.movie_len = movie_len
        self.monitor = monitor
        self.use_scheduler = not scheduler_config is None
        self.scheduler_config = scheduler_config
        self.disc_factor = disc_factor
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        for param in self.clip.parameters():
            param.requires_grad=False
        self.category = {}

    def configure_optimizers(self):
        lr = 1e-1
        params = list()
        params = params + list(self.discriminator.parameters())
        opt = torch.optim.AdamW(params,lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")

            scheduler = [
                {
                    'scheduler':LambdaLR(opt,lr_lambda=scheduler.schedule),
                    'interval':'step',
                    'frequency':1
                }]
            return [opt],scheduler
        return opt

    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del sd[k]
        missing,unexpected = self.load_state_dict(sd,strict=False) if not only_model else self.model.load_state_dict(
            sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self,batch,k):
        x = batch[k]
        # if len(x.shape) == 4:
        #     x = x.unsqueeze(1)
        x = rearrange(x,'b n h w c -> (b n) c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx):
        loss,loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)
        return loss
        
    def validation_step(self,batch,batch_idx):
        _,loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

    def shared_step(self,batch):
        b = batch['reference_image'].shape[0]
        ref_img = self.get_input(batch,'reference_image') # (b n c h w)
        # hdmap = self.get_input(batch,'HDmap') # (b n c h w)
        range_image = self.get_input(batch,'range_image') # (b n c h w)
        # dense_range_image = self.get_input(batch,'dense_range_image') # (b n c h w)
        if '3Dbox' in batch.keys():
            boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()

            boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
            for i in range(len(batch['category'])):
                for j in range(self.movie_len):
                    for k in range(len(batch['category'][0][0])):
                        if not batch['category'][i][j][k] in self.category.keys():
                            self.category[batch['category'][i][j][k]] = self.clip(batch['category'][i][j][k]).cpu().detach().numpy()
                        boxes_category[i][j][k] = self.category[batch['category'][i][j][k]]
            boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
            boxes_category = torch.tensor(boxes_category).to(self.device)
        batch['reference_image'] = ref_img
        if '3Dbox' in batch.keys():
            batch['3Dbox'] = boxes
            batch['category'] = boxes_category
        batch['range_image'] = range_image
        batch = {k:v.to(torch.float32) for k,v in batch.items()}
        out = self.global_condition.get_conditions(batch)
        range_image = out['range_image']
        x = out['ref_image'].to(torch.float32)
        c = {}
        if 'boxes_emb' in out.keys():
            c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        loss,loss_dict = self(x,c,batch,range_image)
        return loss,loss_dict
    
    def forward(self,x_start,cond,batch,range_image_start=None):
        inputs = torch.cat([x_start,range_image_start],dim=1)
        logits_real = self.discriminator(inputs)
        x_start = rearrange(x_start,'(b n) c h w -> b n c h w',n=self.movie_len)
        range_image_start = rearrange(range_image_start,'(b n) c h w -> b n c h w',n=self.movie_len)
        strategy = "batch" if random.randint(0,1) == 0 else 'video'
        # strategy = "batch"
        if strategy == 'batch':
            batch_idx = [i for i in range(x_start.shape[0])]
            sample_idx = random.sample(batch_idx,x_start.shape[0] // 2)
            fake_x_inputs = []
            fake_range_inputs = []
            random_idx = []
            for idx in sample_idx:
                fake_x_inputs.append(x_start[idx:idx+1])
                fake_range_inputs.append(x_start[idx:idx+1])
            for idx in batch_idx:
                if idx in sample_idx:
                    continue
                else:
                    random_idx.append(idx)
            for i in range(len(random_idx)):
                idx,f_idx = random_idx[i],random_idx[(idx+1)%len(random_idx)]
                fake_x_inputs.append(x_start[idx:idx+1])
                fake_range_inputs.append(range_image_start[f_idx:f_idx+1])
            fake_x_inputs = torch.cat(fake_x_inputs,dim=0)
            fake_range_inputs = torch.cat(fake_range_inputs,dim=0)
            fake_x_inputs = rearrange(fake_x_inputs,'b n c h w -> (b n) c h w')
            fake_range_inputs = rearrange(fake_range_inputs,'b n c h w -> (b n) c h w')
            fake_inputs = torch.cat([fake_x_inputs,fake_range_inputs],dim=1)
        else:
            frame_idx = [i for i in range(x_start.shape[1])]
            sample_idx = random.sample(frame_idx,x_start.shape[1]//2)
            fake_x_inputs = []
            fake_range_inputs = []
            random_idx = []
            for idx in sample_idx:
                fake_x_inputs.append(x_start[:,idx:idx+1])
                fake_range_inputs.append(x_start[:,idx:idx+1])
            for idx in frame_idx:
                if idx in sample_idx:
                    continue
                random_idx.append(idx)
            for i in range(len(random_idx)):
                idx,f_idx = random_idx[i],random_idx[(i+1)%len(random_idx)]
                fake_x_inputs.append(x_start[:,idx:idx+1])
                fake_range_inputs.append(range_image_start[:,f_idx:f_idx+1])
            fake_x_inputs = torch.cat(fake_x_inputs,dim=1)
            fake_range_inputs = torch.cat(fake_range_inputs,dim=1)
            fake_x_inputs = rearrange(fake_x_inputs,'b n c h w -> (b n) c h w').contiguous()
            fake_range_inputs = rearrange(fake_range_inputs,'b n c h w -> (b n) c h w').contiguous()
            fake_inputs = torch.cat([fake_x_inputs,fake_range_inputs],dim=1)
        logits_fake = self.discriminator(fake_inputs)
        disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real,logits_fake)
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/disc_loss': d_loss.clone().detach().mean()})
        loss_dict.update({f'{prefix}/logits_real':logits_real.detach().mean()})
        loss_dict.update({f'{prefix}/logits_fake':logits_fake.detach().mean()})
        return d_loss,loss_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/svd_range_ca_dis_pipe_scheduler.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    network = instantiate_from_config(cfg['model']['params']['loss_fn_config']['params']['discriminator_config'])
    print()
    # x = torch.randn(2,10,128,256,3)
    # range_image = torch.randn(2,10,128,256,3)
    x = torch.randn(2,5,4,16,32)
    range_image = torch.randn(2,5,4,16,32)
    cond = torch.randn(2,10,70,1024)
    out = {
           'reference_image':x,
           'range_image':range_image,
        }
    x = rearrange(x,'b n c h w -> (b n) c h w')
    range_image = rearrange(range_image,'b n c h w -> (b n) c h w')
    x_in =  torch.cat([x,range_image],dim=1)
    print(x_in.shape)
    x = network(x_in)
    print(x.shape)