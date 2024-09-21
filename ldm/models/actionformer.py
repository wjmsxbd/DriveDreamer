import os

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange,repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from ldm.modules.diffusionmodules.util import normalization
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img,exists,default,ismap,isimage,mean_flat, count_params, instantiate_from_config,to_cpu
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGuassianDistribution
from ldm.models.autoencoder import AutoencoderKL,VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import CrossAttention
from ldm.modules.diffusionmodules.util import extract_into_tensor
import omegaconf
import time
import concurrent.futures
import threading
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
)
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(4,channels)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            dims=2,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims,channels,self.out_channels,3,padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims,self.out_channels,self.out_channels,3,padding=1)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims,channels,self.out_channels,3,padding=1
            )
        else:
            self.skip_connection = conv_nd(dims,channels,self.out_channels,1)

    def forward(self,x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward,(x),self.parameters(),self.use_checkpoint
        # )
        return self._forward(x)
        
    def _forward(self,x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class GRU_Blocks(nn.Module):
    def __init__(self,
                 hidden_state_channels,
                 in_channels,
                 ):
        super().__init__()
        self.w_xr = nn.Linear(in_channels,hidden_state_channels)
        self.w_xz = nn.Linear(in_channels,hidden_state_channels)
        self.w_hr = nn.Linear(hidden_state_channels,hidden_state_channels)
        self.w_hz = nn.Linear(hidden_state_channels,hidden_state_channels)
        self.w_xh = nn.Linear(in_channels,hidden_state_channels)
        self.w_hh = nn.Linear(hidden_state_channels,hidden_state_channels)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self,input,hidden_state):
        rt = self.sigmoid(self.w_xr(input) + self.w_hr(hidden_state))
        zt = self.sigmoid(self.w_xz(input) + self.w_hz(hidden_state))
        ht = self.tanh(self.w_xh(input) + self.w_hh(rt * hidden_state))
        ht = zt * hidden_state + (1-zt) * ht
        return ht

class ActionEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 trainable=False):
        super().__init__()
        self.module_list = nn.ModuleList([nn.Linear(in_channels,ch[0])])
        self.trainable = trainable
        for i in range(len(ch)-1):
            self.module_list.append(nn.LayerNorm(ch[i]))
            self.module_list.append(nn.Linear(ch[i],ch[i+1]))
            self.module_list.append(nn.SiLU())

    def forward(self,x):
        for layer in self.module_list:
            x = layer(x)
        return x
    
class ActionDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 out_latent_hdmap_channel,
                 out_latent_boxes_channel,
                 num_boxes,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        self.num_boxes = num_boxes
        self.dec_hdmap = nn.Linear(pre_channel,out_latent_hdmap_channel)
        self.dec_boxes = nn.Linear(pre_channel,out_latent_boxes_channel)
        self.dec_range_image = nn.Linear(pre_channel,out_latent_hdmap_channel)
        
    def forward(self,x):
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)
        latent_hdmap = x[:,0]
        latent_boxes = x[:,1:self.num_boxes+1]
        latent_range_image = x[:,-2]

        latent_hdmap = self.dec_hdmap(latent_hdmap)
        latent_boxes = self.dec_boxes(latent_boxes)
        latent_range_image = self.dec_range_image(latent_range_image)
        return [latent_hdmap,latent_boxes,latent_range_image]

class ActionFormer(nn.Module):
    def __init__(self,
                latent_hdmap_dims,
                latent_boxes_dims,
                num_heads,
                dim_head,
                embed_dim,
                z_channels,
                decoder_config,
                gru_blocks_config,
                ckpt_path=None,
                ignore_keys=[],
                modify_keys=[],
                 *args,**kwargs,):
        super().__init__()
        self.linear1 = nn.Linear(latent_hdmap_dims,embed_dim)
        self.linear2 = nn.Linear(latent_boxes_dims,embed_dim)
        self.linear3 = nn.Linear(latent_hdmap_dims,embed_dim)

        self.attn1 = CrossAttention(embed_dim,heads=num_heads,dim_head=dim_head)

        self.linear4 = nn.Linear(embed_dim,embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.attn2 = CrossAttention(embed_dim,heads=num_heads,dim_head=dim_head)
        self.attn3 = CrossAttention(embed_dim,context_dim=embed_dim,heads=num_heads,dim_head=dim_head)
        self.linear5 = nn.Linear(embed_dim,2*z_channels)
        self.gru_block = instantiate_from_config(gru_blocks_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.z_channels = z_channels
        if not ckpt_path is None:
            self.init_from_ckpt(ckpt_path,ignore_keys,modify_keys)

    def init_from_ckpt(self,ckpt_path,ignore_keys,modify_keys):
        model = torch.load(ckpt_path,map_location='cpu')['state_dict']
        keys = list(model.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startwith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del model[k]
            for ik in modify_keys:
                if k.startswith(ik):
                    modify_k = k[14:]
                    model[modify_k] = model[k]
                    del(model[k])
                    print("Modify Key {} to {} in state_dict".format(k,modify_k))
        missing,unexpected = self.load_state_dict(model,strict=False)

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (c h w)').unsqueeze(1).contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (c h w)').unsqueeze(1).contiguous()

        latent_hdmap = self.linear1(latent_hdmap)
        latent_boxes = self.linear2(latent_boxes)
        latent_dense_range_image = self.linear3(latent_dense_range_image)

        latent = torch.cat([latent_hdmap,latent_boxes,latent_dense_range_image],dim=1)

        x = self.attn1(self.norm_latent(latent))

        x = self.linear4(self.norm(x))

        h = []
        for timestamp in range(action.shape[1]):
            if timestamp == 0:
                latent_x = torch.cat([x,torch.zeros_like(action[:,0:1])],dim=1)
                h.append(self.decoder(latent_x))
            else:
                s = x
                s = self.attn2(self.norm2(s))
                s = self.attn3(self.norm3(s),action[:,timestamp-1:timestamp])
                moments = self.linear5(s)
                b,n,c = moments.shape[0],moments.shape[1],moments.shape[2]
                moments = moments.reshape(-1,c)
                posterior = DiagonalGuassianDistribution(moments)
                s = posterior.sample()
                s = s.reshape(b,n,self.z_channels)
                x = self.gru_block(s,x)
                latent_x = torch.cat([x,action[:,timestamp-1:timestamp]],dim=1)
                h.append(self.decoder(latent_x))
        return h

class PipeLineActionFormer(pl.LightningModule):
    def __init__(self,actionformer_config,global_condition_config,cond_stage_config,movie_len,predict,use_scheduler=False,scheduler_config=None,monitor='val/loss',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.action_former = instantiate_from_config(actionformer_config)
        self.global_condition = instantiate_from_config(global_condition_config)
        self.clip = instantiate_from_config(cond_stage_config)
        self.clip = self.clip.eval()
        self.movie_len = movie_len
        self.monitor = monitor
        self.use_scheduler = not scheduler_config is None
        self.scheduler_config = scheduler_config
        self.predict = predict
        for param in self.clip.parameters():
            param.requires_grad=False

    def configure_optimizers(self):
        lr = 1e-5
        params = list()
        params = params + list(self.action_former.parameters())
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
        hdmap = self.get_input(batch,'HDmap') # (b n c h w)
        range_image = self.get_input(batch,'range_image') # (b n c h w)
        dense_range_image = self.get_input(batch,'dense_range_image') # (b n c h w)
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()

        boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
        for i in range(len(batch['category'])):
            for j in range(self.movie_len):
                for k in range(len(batch['category'][0][0])):
                    boxes_category[i][j][k] = self.clip(batch['category'][i][j][k]).cpu().detach().numpy()
        boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
        boxes_category = torch.tensor(boxes_category).to(self.device)
        text = np.zeros((len(batch['text']),self.movie_len,768))
        for i in range(len(batch['text'])):
            for j in range(self.movie_len):
                text[i,j] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
        text = text.reshape(b*self.movie_len,1,768)
        text = torch.tensor(text).to(self.device)
        batch['reference_image'] = ref_img
        batch['HDmap'] = hdmap
        batch['3Dbox'] = boxes
        batch['category'] = boxes_category
        batch['text'] = text
        batch['range_image'] = range_image
        batch['dense_range_image'] = dense_range_image
        if self.predict:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        range_image = out['range_image']
        x = out['ref_image'].to(torch.float32)
        c = {}
        c['hdmap'] = out['hdmap'].to(torch.float32)
        c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        c['text_emb'] = out['text_emb'].to(torch.float32)
        c['dense_range_image'] = out['dense_range_image']
        c['origin_range_image'] = range_image
        if self.predict:
            c['actions'] = out['actions']
        loss,loss_dict = self(x,range_image,c)
        return loss,loss_dict
    
    def forward(self,x,range_image,c):
        return self.p_losses(x,range_image,c)
    
    def apply_model(self,cond):
        boxes_emb = cond['boxes_emb']
        dense_range_image = cond['dense_range_image']
        b = boxes_emb.shape[0] // self.movie_len
        boxes_emb = boxes_emb.reshape((b,self.movie_len)+boxes_emb.shape[1:])
        boxes_emb = boxes_emb[:,0]
        hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
        actions = cond['actions']
        dense_range_image = dense_range_image.reshape((b,self.movie_len)+dense_range_image.shape[1:])[:,0]
        h = self.action_former(hdmap,boxes_emb,actions,dense_range_image)
        latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
        latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
        latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
        return latent_hdmap,latent_boxes,latent_dense_range_image
        

    def p_losses(self,x_start,range_image_start,cond):
        latent_hdmap_recon,latent_boxes_recon,latent_dense_range_image_recon = self.apply_model(cond)
        latent_hdmap = cond['hdmap']
        latent_boxes = cond['boxes_emb']
        latent_dense_range_image = cond['dense_range_image']
        loss_hdmap = torch.abs(latent_hdmap - latent_hdmap_recon).mean()
        loss_boxes = torch.abs(latent_boxes - latent_boxes_recon).mean()
        loss_dense = torch.abs(latent_dense_range_image - latent_dense_range_image_recon).mean()
        loss = loss_hdmap / loss_hdmap.detach() + loss_boxes / loss_boxes.detach() + loss_dense / loss_dense.detach()
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': loss_hdmap + loss_dense + loss_boxes})
        loss_dict.update({f'{prefix}/loss_hdmap':loss_hdmap})
        loss_dict.update({f'{prefix}/loss_boxes':loss_boxes})
        loss_dict.update({f'{prefix}/loss_dense':loss_dense})
        return loss,loss_dict
    

class ActionDecoder2(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 out_latent_hdmap_channel,
                 out_latent_boxes_channel,
                 num_boxes,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        self.num_boxes = num_boxes
        self.dec_hdmap = nn.Linear(pre_channel,out_latent_hdmap_channel)
        self.dec_boxes = nn.Linear(pre_channel,out_latent_boxes_channel)
        self.dec_range_image = nn.Linear(pre_channel,out_latent_hdmap_channel)
        
    def forward(self,x):
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)
        latent_hdmap = x[:,:512]
        latent_boxes = x[:,512:self.num_boxes+512]
        latent_range_image = x[:,512+self.num_boxes:-1]

        latent_hdmap = self.dec_hdmap(latent_hdmap)
        latent_boxes = self.dec_boxes(latent_boxes)
        latent_range_image = self.dec_range_image(latent_range_image)
        latent_hdmap = rearrange(latent_hdmap,'b n c -> b c n').contiguous()
        latent_range_image = rearrange(latent_range_image,'b n c -> b c n').contiguous()
        return [latent_hdmap,latent_boxes,latent_range_image]

class ActionFormer2(nn.Module):
    def __init__(self,
                latent_hdmap_dims,
                latent_boxes_dims,
                num_heads,
                dim_head,
                embed_dim,
                z_channels,
                decoder_config,
                gru_blocks_config,
                ckpt_path=None,
                ignore_keys=[],
                modify_keys=[],
                 *args,**kwargs,):
        super().__init__()
        self.linear1 = nn.Linear(latent_hdmap_dims,embed_dim)
        self.linear2 = nn.Linear(latent_boxes_dims,embed_dim)
        self.linear3 = nn.Linear(latent_hdmap_dims,embed_dim)

        self.attn1 = CrossAttention(embed_dim,heads=num_heads,dim_head=dim_head)

        self.linear4 = nn.Linear(embed_dim,embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.attn2 = CrossAttention(embed_dim,heads=num_heads,dim_head=dim_head)
        self.attn3 = CrossAttention(embed_dim,context_dim=embed_dim,heads=num_heads,dim_head=dim_head)
        self.linear5 = nn.Linear(embed_dim,2*z_channels)
        self.gru_block = instantiate_from_config(gru_blocks_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.z_channels = z_channels
        if not ckpt_path is None:
            self.init_from_ckpt(ckpt_path,ignore_keys,modify_keys)

    def init_from_ckpt(self,ckpt_path,ignore_keys,modify_keys):
        model = torch.load(ckpt_path,map_location='cpu')['state_dict']
        keys = list(model.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startwith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del model[k]
            for ik in modify_keys:
                if k.startswith(ik):
                    modify_k = k[14:]
                    model[modify_k] = model[k]
                    del(model[k])
                    print("Modify Key {} to {} in state_dict".format(k,modify_k))
        missing,unexpected = self.load_state_dict(model,strict=False)

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (h w) c').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (h w) c').contiguous()

        latent_hdmap = self.linear1(latent_hdmap)
        latent_boxes = self.linear2(latent_boxes)
        latent_dense_range_image = self.linear3(latent_dense_range_image)

        latent = torch.cat([latent_hdmap,latent_boxes,latent_dense_range_image],dim=1)

        x = self.attn1(self.norm_latent(latent))

        x = self.linear4(self.norm(x))

        h = []
        for timestamp in range(action.shape[1]):
            if timestamp == 0:
                latent_x = torch.cat([x,torch.zeros_like(action[:,0:1])],dim=1)
                h.append(self.decoder(latent_x))
            else:
                s = x
                s = self.attn2(self.norm2(s))
                s = self.attn3(self.norm3(s),action[:,timestamp-1:timestamp])
                moments = self.linear5(s)
                b,n,c = moments.shape[0],moments.shape[1],moments.shape[2]
                moments = moments.reshape(-1,c)
                posterior = DiagonalGuassianDistribution(moments)
                s = posterior.sample()
                s = s.reshape(b,n,self.z_channels)
                x = self.gru_block(s,x)
                latent_x = torch.cat([x,action[:,timestamp-1:timestamp]],dim=1)
                h.append(self.decoder(latent_x))
        return h

class ActionDecoder3(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 out_latent_hdmap_channel,
                 out_latent_boxes_channel,
                 latent_boxes_channel,
                 embed_boxes_dim,
                 num_boxes,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        self.num_boxes = num_boxes
        self.embed_boxes_dim = embed_boxes_dim
        self.dec_hdmap = nn.Linear(pre_channel,out_latent_hdmap_channel)
        self.dec_boxes = nn.Linear(pre_channel,latent_boxes_channel)
        self.dec_boxes2 = nn.Linear(embed_boxes_dim,out_latent_boxes_channel)
        self.dec_range_image = nn.Linear(pre_channel,out_latent_hdmap_channel)
        
    def forward(self,x):
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)

        latent_hdmap = self.dec_hdmap(x)
        latent_boxes = self.dec_boxes(x)
        latent_boxes = rearrange(latent_boxes,'b l (n c) -> b l n c',n=self.num_boxes,c=self.embed_boxes_dim)[:,0]
        latent_boxes = self.dec_boxes2(latent_boxes)
        latent_range_image = self.dec_range_image(x)
        return [latent_hdmap,latent_boxes,latent_range_image]

class ActionFormer3(nn.Module):
    def __init__(self,
                latent_dims,
                box_dims,
                box_embed_dims,
                num_heads,
                dim_head,
                embed_dim,
                action_dims,
                z_channels,
                decoder_config,
                gru_blocks_config,
                ckpt_path=None,
                ignore_keys=[],
                modify_keys=[],
                 *args,**kwargs,):
        super().__init__()
        self.box_project = nn.Linear(box_dims,box_embed_dims)
        self.attn1 = CrossAttention(latent_dims,heads=num_heads,dim_head=dim_head)

        self.linear = nn.Linear(latent_dims,embed_dim)
        self.norm_latent = nn.LayerNorm(latent_dims)
        self.norm = nn.LayerNorm(latent_dims)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.attn2 = CrossAttention(embed_dim,heads=num_heads,dim_head=dim_head)
        self.attn3 = CrossAttention(embed_dim,context_dim=action_dims,heads=num_heads,dim_head=dim_head)
        self.linear5 = nn.Linear(embed_dim,2*z_channels)
        self.gru_block = nn.GRU(input_size=gru_blocks_config['params']['in_channels'],hidden_size=gru_blocks_config['params']['hidden_state_channels'],num_layers=1)
        self.decoder = instantiate_from_config(decoder_config)
        self.z_channels = z_channels
        if not ckpt_path is None:
            self.init_from_ckpt(ckpt_path,ignore_keys,modify_keys)

    def init_from_ckpt(self,ckpt_path,ignore_keys,modify_keys):
        model = torch.load(ckpt_path,map_location='cpu')['state_dict']
        keys = list(model.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startwith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del model[k]
            for ik in modify_keys:
                if k.startswith(ik):
                    modify_k = k[14:]
                    model[modify_k] = model[k]
                    del(model[k])
                    print("Modify Key {} to {} in state_dict".format(k,modify_k))
        missing,unexpected = self.load_state_dict(model,strict=False)

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (c h w)').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (c h w)').contiguous()
        latent_boxes = self.box_project(latent_boxes)
        latent_boxes = rearrange(latent_boxes,'b n c -> b (n c)').contiguous()
        latent = torch.cat([latent_hdmap,latent_boxes,latent_dense_range_image],dim=1).unsqueeze(1)

        x = self.attn1(self.norm_latent(latent))

        x = self.linear(self.norm(x))

        h = []
        for timestamp in range(action.shape[1]):
            if timestamp == 0:
                latent_x = torch.cat([x,torch.zeros_like(action[:,0:1])],dim=-1)
                h.append(self.decoder(latent_x))
            else:
                s = x
                s = self.attn2(self.norm2(s))
                s = self.attn3(self.norm3(s),action[:,timestamp-1:timestamp])
                moments = self.linear5(s)
                b,n,c = moments.shape[0],moments.shape[1],moments.shape[2]
                moments = moments.reshape(-1,c)
                posterior = DiagonalGuassianDistribution(moments)
                s = posterior.sample()
                s = s.reshape(b,n,self.z_channels)
                s = rearrange(s,'b n c -> n b c').contiguous()
                x = rearrange(x,'b n c -> n b c').contiguous()
                x,_ = self.gru_block(s,x)
                x = rearrange(x,'n b c -> b n c').contiguous()
                latent_x = torch.cat([x,action[:,timestamp-1:timestamp]],dim=-1)
                h.append(self.decoder(latent_x))
        return h

class ActionDecoder4(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 out_latent_hdmap_channel,
                 out_latent_boxes_channel,
                 num_boxes,
                 height,width,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        self.num_boxes = num_boxes
        self.height = height
        self.width = width
        self.dec_hdmap = nn.Linear(pre_channel,out_latent_hdmap_channel)
        self.dec_boxes = nn.Linear(pre_channel,num_boxes*16)
        self.norm_boxes = nn.LayerNorm(16)
        self.dec_boxes2 = nn.Linear(16,out_latent_boxes_channel)
        self.dec_range_image = nn.Linear(pre_channel,out_latent_hdmap_channel)
        
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)
        x = x[:,0]
        s = x
        # x = rearrange(x,'b (h w c) -> b h w c',c=4,h=self.height,w=self.width)
        latent_hdmap = self.dec_hdmap(x)

        latent_boxes = self.dec_boxes(s)
        latent_boxes = rearrange(latent_boxes,'b (n c) -> b n c',n=self.num_boxes,c=16).contiguous()
        latent_boxes = self.dec_boxes2(self.norm_boxes(latent_boxes))

        latent_range_image = self.dec_range_image(x)
        # latent_hdmap = rearrange(latent_hdmap,'b h w c -> b c h w').contiguous()
        # latent_range_image = rearrange(latent_range_image,'b h w c -> b c h w').contiguous()
        return [latent_hdmap,latent_boxes,latent_range_image]


class ActionFormer4(nn.Module):
    def __init__(self,latent_range_image_dims,latent_hdmap_dims,latent_boxes_dims,heads,dim_head,latent_dims,
                 embed_dims,action_dims,gru_blocks_config,decoder_config):
        super().__init__()
        self.attn_r2h = CrossAttention(latent_range_image_dims,latent_hdmap_dims,heads=heads,dim_head=dim_head)
        self.attn_r2b = CrossAttention(latent_range_image_dims,latent_boxes_dims,heads=heads,dim_head=dim_head)
        self.norm_r2h = nn.LayerNorm(latent_range_image_dims)
        self.norm_r2b = nn.LayerNorm(latent_range_image_dims)
        self.attn1 = CrossAttention(latent_range_image_dims,heads=heads,dim_head=dim_head)
        self.attn2 = CrossAttention(latent_range_image_dims,heads=heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(latent_range_image_dims)
        self.norm2 = nn.LayerNorm(latent_range_image_dims)
        self.attn3 = CrossAttention(latent_range_image_dims,latent_range_image_dims,heads=heads,dim_head=dim_head)
        self.norm3 = nn.LayerNorm(latent_range_image_dims)
        self.proj = nn.Linear(latent_dims,1)

        self.attn4 = CrossAttention(embed_dims,heads=heads,dim_head=dim_head)
        self.norm4 = nn.LayerNorm(embed_dims)

        self.attn5 = CrossAttention(embed_dims,action_dims,heads=heads,dim_head=dim_head)
        self.norm5_1 = nn.LayerNorm(embed_dims)
        self.norm5_2 = nn.LayerNorm(action_dims)

        self.gru_block = nn.GRU(input_size=gru_blocks_config['params']['in_channels'],hidden_size=gru_blocks_config['params']['hidden_state_channels'],num_layers=1)
        self.decoder = instantiate_from_config(decoder_config)


    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        b,c,h,w = latent_hdmap.shape
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (h w) c').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (h w) c').contiguous()
        r2h = self.attn_r2h(latent_dense_range_image,latent_hdmap)

        r2b = self.attn_r2b(latent_dense_range_image,latent_boxes)
        r2h = self.attn1(self.norm_r2h(r2h))
        r2b = self.attn2(self.norm_r2b(r2b))
        x = latent_dense_range_image + self.attn3(self.norm1(r2h),self.norm2(r2b))
        # x = self.proj(self.norm3(x))
        x = x.reshape(b,-1)
        h = []
        for timestamp in range(action.shape[1]):
            if timestamp == 0:
                latent_x = torch.cat([x,torch.zeros_like(action[:,0])],dim=-1)
                h.append(self.decoder(latent_x))
            else:
                s = x
                s = s.unsqueeze(1)
                x = x.unsqueeze(1)
                s = self.attn4(self.norm4(s))
                s = self.attn5(self.norm5_1(s),self.norm5_2(action[:,timestamp-1:timestamp]))
                s = rearrange(s,'b n c -> n b c').contiguous()
                x = rearrange(x,'b n c -> n b c').contiguous()
                x,_ = self.gru_block(s,x)
                x = x[0]
                latent_x = torch.cat([x,action[:,timestamp-1]],dim=-1)
                h.append(self.decoder(latent_x))
        return h

class ActionDecoder5(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        
    def forward(self,x,actions):
        x = torch.cat([x,actions],dim=-1)
        x = x.unsqueeze(1)
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)
        x = x[:,0]
        return x

class ActionFormer5(nn.Module):
    def __init__(self,latent_bev_dims,flatten_dims,embed_dims,action_dims,gru_blocks_config,heads,dim_head,decoder_config):
        super().__init__()
        self.latent_bev_dims = latent_bev_dims
        self.action_dims = action_dims
        self.linear = nn.Linear(flatten_dims,embed_dims)
        self.silu = nn.SiLU()
        self.attn1 = CrossAttention(query_dim=embed_dims,heads=heads,dim_head=dim_head)
        self.attn2 = CrossAttention(query_dim=embed_dims,context_dim=action_dims,heads=heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(action_dims)
        self.gru_block = nn.GRU(input_size=gru_blocks_config['params']['in_channels'],hidden_size=gru_blocks_config['params']['hidden_state_channels'],num_layers=1)
        self.decoder = instantiate_from_config(decoder_config)

    def forward(self,latent_bev,actions):
        b,c,h,w = latent_bev.shape
        latent_bev = rearrange(latent_bev,'b c h w -> b (h w) c').contiguous()
        x = latent_bev
        x = x.reshape(b,-1)
        x = self.linear(x)
        # x = self.silu(x)
        h = []
        for timestamp in range(actions.shape[1]):
            if timestamp == 0:
                h.append(self.decoder(x,torch.zeros_like(actions[:,0])))
            else:
                s = x
                s = s.unsqueeze(1)
                x = x.unsqueeze(1)
                s = self.attn1(self.norm1(s))
                s = self.attn2(self.norm2(s),self.norm3(actions[:,timestamp-1:timestamp]))
                s = rearrange(s,'b n c -> n b c').contiguous()
                x = rearrange(x,'b n c -> n b c').contiguous()
                x,_ = self.gru_block(s,x)
                x = x[0]
                h.append(self.decoder(x,actions[:,timestamp-1]))
        return h

class ActionDecoder6(nn.Module):
    def __init__(self,
                 in_channels,
                 ch,
                 num_heads=8,
                 dim_head=64):
        super().__init__()
        self.attn1 = CrossAttention(in_channels,heads=num_heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(in_channels)
        self.linear = nn.ModuleList()
        pre_channel = in_channels
        for i in range(len(ch)):
            if i!=0:
                self.linear.append(nn.LayerNorm(pre_channel))
            self.linear.append(nn.Linear(pre_channel,ch[i]))
            self.linear.append(nn.SiLU())
            pre_channel = ch[i]
        

    def forward(self,x,actions):
        x = torch.cat([x,actions],dim=-1)
        x = x.unsqueeze(1)
        x = self.attn1(self.norm1(x))
        for layer in self.linear:
            x = layer(x)
        x = x[:,0]
        return x

class ActionFormer6(nn.Module):
    def __init__(self,latent_bev_dims,flatten_dims,embed_dims,action_dims,gru_blocks_config,heads,dim_head,decoder_config,dropout=0.0):
        super().__init__()
        self.latent_bev_dims = latent_bev_dims
        self.action_dims = action_dims
        self.conv1 = ResBlock(latent_bev_dims,dropout)
        self.conv2 = ResBlock(latent_bev_dims,dropout)
        self.linear = nn.Linear(flatten_dims,embed_dims)
        self.attn1 = CrossAttention(query_dim=embed_dims,heads=heads,dim_head=dim_head)
        self.attn2 = CrossAttention(query_dim=embed_dims,context_dim=action_dims,heads=heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(action_dims)
        self.gru_block = nn.GRU(input_size=gru_blocks_config['params']['in_channels'],hidden_size=gru_blocks_config['params']['hidden_state_channels'],num_layers=1)
        self.decoder = instantiate_from_config(decoder_config)

    def forward(self,latent_bev,actions):
        b,c,h,w = latent_bev.shape
        # latent_bev = rearrange(latent_bev,'b c h w -> b (h w) c').contiguous()
        x = latent_bev
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(b,-1)
        x = self.linear(x)
        # x = self.silu(x)
        h = []
        for timestamp in range(actions.shape[1]):
            if timestamp == 0:
                h.append(self.decoder(x,torch.zeros_like(actions[:,0])))
            else:
                s = x
                s = s.unsqueeze(1)
                x = x.unsqueeze(1)
                s = self.attn1(self.norm1(s))
                s = self.attn2(self.norm2(s),self.norm3(actions[:,timestamp-1:timestamp]))
                s = rearrange(s,'b n c -> n b c').contiguous()
                x = rearrange(x,'b n c -> n b c').contiguous()
                x,_ = self.gru_block(s,x)
                x = x[0]
                h.append(self.decoder(x,actions[:,timestamp-1]))
        return h

class PipeLineActionFormer5(pl.LightningModule):
    def __init__(self,actionformer_config,global_condition_config,movie_len,predict,use_scheduler=False,scheduler_config=None,monitor='val/loss',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.action_former = instantiate_from_config(actionformer_config)
        self.global_condition = instantiate_from_config(global_condition_config)

        self.movie_len = movie_len
        self.monitor = monitor
        self.use_scheduler = not scheduler_config is None
        self.scheduler_config = scheduler_config
        self.predict = predict

    def configure_optimizers(self):
        lr = 1e-5
        params = list()
        params = params + list(self.action_former.parameters()) + list(self.global_condition.get_parameters())
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
        b = batch['bev_images'].shape[0]
        bev_imgs = self.get_input(batch,'bev_images')
        batch['reference_image'] = self.get_input(batch,'reference_image')
        batch['range_image'] = self.get_input(batch,'range_image')
        batch['bev_images'] = bev_imgs
        if self.predict:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        bev_image = out['bev_images']
        actions = out['actions']
        loss,loss_dict = self(bev_image,actions)
        return loss,loss_dict
    
    def forward(self,bev_image,actions):
        return self.p_losses(bev_image,actions)
    
    def apply_model(self,bev_image,actions):
        b = actions.shape[0]
        x = bev_image.reshape((b,self.movie_len)+bev_image.shape[1:])
        x = x[:,0]
        h = self.action_former(x,actions)
        bev_rec = torch.stack([h[id] for id in range(len(h))]).reshape(bev_image.shape)
        return bev_rec
        

    def p_losses(self,bev_image,actions):
        bev_rec = self.apply_model(bev_image,actions)
        loss_bev = torch.abs(bev_rec - bev_image).mean()
        
        loss = loss_bev
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': loss_bev})
        return loss,loss_dict
    
    def log_video(self,batch,N=8,n_row=4,**kwargs):
        log = dict()
        b = batch['reference_image'].shape[0]
        bev_imgs = self.get_input(batch,'bev_images')
        batch['reference_image'] = self.get_input(batch,'reference_image')
        batch['range_image'] = self.get_input(batch,'range_image')
        batch['bev_images'] = bev_imgs
        if self.predict:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        bev_image = out['bev_images']
        actions = out['actions']
        bev_rec = self.apply_model(bev_image,actions)
        bev_rec = self.global_condition.decode_first_stage_interface('reference_image',bev_rec)

        log['inputs'] = bev_imgs.reshape((b,self.movie_len)+bev_imgs.shape[1:])
        log['reconstructions'] = bev_rec.reshape((b,self.movie_len)+bev_rec.shape[1:])
        return log



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/actionformer6_train.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    actionformer = instantiate_from_config(cfg['model']['params']['actionformer_config'])
    # latent_hdmap = torch.randn((2,4,16,32))
    # latent_boxes = torch.randn((2,70,768))
    latent_bev = torch.randn(2,4,64,64)
    action = torch.randn(2,5,128)
    # dense_range_image = torch.randn((2,4,16,32))
    actionformer(latent_bev,action)
