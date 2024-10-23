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
import torch.nn.functional as F
import omegaconf
import copy
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
    
class ActionEncoder2(nn.Module):
    def __init__(self,action_dims,latent_dims,trainable=True):
        super().__init__()
        self.trainable = trainable
        self.speed_enc = nn.Sequential(
            nn.Linear(3,action_dims),
            nn.ReLU(True),
            nn.Linear(action_dims,action_dims),
            nn.ReLU(True)
        )
        self.rotation_enc = nn.Sequential(
            nn.Linear(6,action_dims),
            nn.ReLU(True),
            nn.Linear(action_dims,action_dims),
            nn.ReLU(True),
        )
        self.command_enc = nn.Sequential(
            nn.Linear(3,action_dims),
            nn.ReLU(True),
            nn.Linear(action_dims,action_dims),
            nn.ReLU(True),
        )
        self.acc_enc = nn.Sequential(
            nn.Linear(3,action_dims),
            nn.ReLU(),
            nn.Linear(action_dims,action_dims),
            nn.ReLU(True),
        )
        self.trajectory_enc = nn.Sequential(
            nn.Linear(3,action_dims),
            nn.ReLU(True),
            nn.Linear(action_dims,action_dims),
        )
        self.linear = nn.Sequential(
            nn.Linear(7*action_dims,latent_dims),
            nn.LeakyReLU(True)
        )

    def forward(self,x):
        velocity = x[:,:,:3]
        acc = x[:,:,3:6]
        oritation = x[:,:,6:12]
        command = x[:,:,12:15]
        cam_translation = x[:,:,15:18]
        cam_rotation = x[:,:,18:24]
        trajectory = x[:,:,24:]
        
        velocity = self.speed_enc(velocity)
        acc = self.acc_enc(acc)
        oritation = self.rotation_enc(oritation)
        command = self.command_enc(command)
        cam_translation = self.trajectory_enc(cam_translation)
        cam_rotation = self.rotation_enc(cam_rotation)
        trajectory = self.trajectory_enc(trajectory)
        actions = torch.cat([velocity,acc,oritation,command,cam_translation,cam_rotation,trajectory],dim=-1)
        actions = self.linear(actions)
        return actions

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
        if x.shape[1] == self.num_boxes + 3:
            latent_range_image = x[:,-2]

        latent_hdmap = self.dec_hdmap(latent_hdmap)
        latent_boxes = self.dec_boxes(latent_boxes)
        if x.shape[1] == self.num_boxes + 3:
            latent_range_image = self.dec_range_image(latent_range_image)
            return [latent_hdmap,latent_boxes,latent_range_image]
        return [latent_hdmap,latent_boxes]

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

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image=None):
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (c h w)').unsqueeze(1).contiguous()
        if not latent_dense_range_image is None:
            latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (c h w)').unsqueeze(1).contiguous()
            latent_dense_range_image = self.linear3(latent_dense_range_image)

        latent_hdmap = self.linear1(latent_hdmap)
        latent_boxes = self.linear2(latent_boxes)

        if not latent_dense_range_image is None:
            latent = torch.cat([latent_hdmap,latent_boxes,latent_dense_range_image],dim=1)
        else:
            latent = torch.cat([latent_hdmap,latent_boxes],dim=1)

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

class Box_Decoder(nn.Module):
    def __init__(self,in_channels,ch=[]):
        super().__init__()
        self.linear1 = nn.Linear(in_channels,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,32)
        self.linear4 = nn.Linear(32,16)
        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.norm2 = torch.nn.LayerNorm(128)
        self.norm3 = torch.nn.LayerNorm(64)
        self.norm4 = torch.nn.LayerNorm(32)

    def forward(self,x):
        x = self.linear1(self.norm1(x))
        x = self.linear2(self.norm2(x))
        x = self.linear3(self.norm3(x))
        x = self.linear4(self.norm4(x))
        return x


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
        self.category = {}

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
        x = x.float()
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
        # range_image = self.get_input(batch,'range_image') # (b n c h w)
        # dense_range_image = self.get_input(batch,'dense_range_image') # (b n c h w)
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
        # text = np.zeros((len(batch['text']),self.movie_len,768))
        # for i in range(len(batch['text'])):
        #     for j in range(self.movie_len):
        #         text[i,j] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
        # text = text.reshape(b*self.movie_len,1,768)
        # text = torch.tensor(text).to(self.device)
        batch['reference_image'] = ref_img
        batch['HDmap'] = hdmap
        batch['3Dbox'] = boxes
        batch['category'] = boxes_category
        # batch['text'] = text
        # batch['range_image'] = range_image
        # batch['dense_range_image'] = dense_range_image
        if self.predict:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        # range_image = out['range_image']
        x = out['ref_image'].to(torch.float32)
        c = {}
        c['hdmap'] = out['hdmap'].to(torch.float32)
        c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        # c['text_emb'] = out['text_emb'].to(torch.float32)
        # c['dense_range_image'] = out['dense_range_image']
        # c['origin_range_image'] = range_image
        if self.predict:
            c['actions'] = out['actions']
        # loss,loss_dict = self(x,range_image,c)
        loss,loss_dict = self(x,c,batch)
        return loss,loss_dict
    
    def forward(self,x,c,batch,range_image=None):
        if range_image is None:
            return self.p_losses(x,c,batch)
        else:
            return self.p_losses(x,c,batch,range_image)
    
    def apply_model(self,cond):
        boxes_emb = cond['boxes_emb']
        # dense_range_image = cond['dense_range_image']
        b = boxes_emb.shape[0] // self.movie_len
        boxes_emb = boxes_emb.reshape((b,self.movie_len)+boxes_emb.shape[1:])
        boxes_emb = boxes_emb[:,0]
        hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
        actions = cond['actions']
        # dense_range_image = dense_range_image.reshape((b,self.movie_len)+dense_range_image.shape[1:])[:,0]
        # h = self.action_former(hdmap,boxes_emb,actions,dense_range_image)
        h = self.action_former(hdmap,boxes_emb,actions)
        latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
        latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
        # latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
        # return latent_hdmap,latent_boxes,latent_dense_range_image
        return latent_hdmap,latent_boxes

    @torch.no_grad() 
    def log_images(self,batch,only_inputs=False,**kwargs):
        log = dict()
        b = batch['reference_image'].shape[0]
        ref_img = self.get_input(batch,'reference_image') # (b n c h w)
        hdmap = self.get_input(batch,'HDmap') # (b n c h w)
        log['inputs'] = hdmap
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
        batch['HDmap'] = hdmap
        batch['3Dbox'] = boxes
        batch['category'] = boxes_category
        # batch['text'] = text
        # batch['range_image'] = range_image
        # batch['dense_range_image'] = dense_range_image
        if self.predict:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        # range_image = out['range_image']
        x = out['ref_image'].to(torch.float32)
        c = {}
        c['hdmap'] = out['hdmap'].to(torch.float32)
        c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        # c['text_emb'] = out['text_emb'].to(torch.float32)
        # c['dense_range_image'] = out['dense_range_image']
        # c['origin_range_image'] = range_image
        if self.predict:
            c['actions'] = out['actions']
        latent_hdmap,latent_boxes = self.apply_model(c)
        latent_hdmap = self.global_condition.decode_first_stage_interface('reference_image',latent_hdmap)
        log['reconstructions'] = latent_hdmap
        return log

    def p_losses(self,x_start,cond,batch,range_image_start=None):
        # latent_hdmap_recon,latent_boxes_recon,latent_dense_range_image_recon = self.apply_model(cond)
        latent_hdmap_recon,latent_boxes_recon = self.apply_model(cond)
        latent_hdmap = cond['hdmap']
        latent_boxes = cond['boxes_emb']
        # latent_dense_range_image = cond['dense_range_image']
        # rec_hdmap = self.global_condition.decode_first_stage_interface('reference_image',latent_hdmap_recon)
        # rec_box = self.box_decoder(latent_boxes_recon)
        loss_hdmap = torch.abs(latent_hdmap - latent_hdmap_recon).mean()
        loss_boxes = torch.abs(latent_boxes_recon - latent_boxes).mean()
        # loss_dense = torch.abs(latent_dense_range_image - latent_dense_range_image_recon).mean()
        # loss = loss_hdmap / loss_hdmap.detach() + loss_boxes / loss_boxes.detach() + loss_dense / loss_dense.detach()
        loss = loss_hdmap / loss_hdmap.detach() + loss_boxes / loss_boxes.detach()
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': loss_hdmap + loss_boxes})
        # loss_dict.update({f'{prefix}/loss_simple': loss_hdmap + loss_boxes + loss_dense})
        loss_dict.update({f'{prefix}/loss_hdmap':loss_hdmap})
        loss_dict.update({f'{prefix}/loss_boxes':loss_boxes})
        # loss_dict.update({f'{prefix}/loss_dense':loss_dense})
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
        latent_dense_range_image = torch.cat([latent_dense_range_image,latent_hdmap],dim=1)
        # r2h = self.attn_r2h(latent_dense_range_image,latent_hdmap)

        r2b = self.attn_r2b(latent_dense_range_image,latent_boxes)
        # r2h = self.attn1(self.norm_r2h(r2h))
        r2b = self.attn2(self.norm_r2b(r2b))
        x = latent_dense_range_image + self.attn3(self.norm1(latent_dense_range_image),self.norm2(r2b))
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
        x = x.float()
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

class ActionFormer7(nn.Module):
    def __init__(self,latent_range_image_dims,latent_hdmap_dims,latent_boxes_dims,heads,dim_head,latent_dims,
                 embed_dims,action_dims,gru_blocks_config,decoder_config,z_channels):
        super().__init__()
        self.attn_r2b = CrossAttention(latent_range_image_dims,latent_boxes_dims,heads=heads,dim_head=dim_head)
        self.self_attn = CrossAttention(latent_range_image_dims,heads=heads,dim_head=dim_head)
        self.norm_pre1 = nn.LayerNorm(latent_range_image_dims)
        self.norm_pre2 = nn.LayerNorm(latent_range_image_dims)
        self.attn1 = CrossAttention(latent_range_image_dims,heads=heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(latent_range_image_dims)
        
        self.linear4 = nn.Linear(latent_dims,embed_dims)
        self.norm2 = nn.LayerNorm(latent_dims)
        self.z_channels = z_channels
        self.linear5 = nn.Linear(embed_dims,2*z_channels)

        self.attn4 = CrossAttention(embed_dims,heads=heads,dim_head=dim_head)
        self.norm4 = nn.LayerNorm(embed_dims)

        self.attn5 = CrossAttention(embed_dims,action_dims,heads=heads,dim_head=dim_head)
        self.norm5_1 = nn.LayerNorm(embed_dims)

        self.gru_block = nn.GRU(input_size=gru_blocks_config['params']['in_channels'],hidden_size=gru_blocks_config['params']['hidden_state_channels'],num_layers=1)
        self.decoder = instantiate_from_config(decoder_config)


    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        b,c,h,w = latent_hdmap.shape
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (h w) c').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (h w) c').contiguous()
        latent_z = torch.cat([latent_dense_range_image,latent_hdmap],dim=1)
        latent_z = self.self_attn(self.norm_pre1(latent_z)) + latent_z
        latent_z = self.attn_r2b(self.norm_pre2(latent_z),latent_boxes) + latent_z
        latent_z = self.attn1(self.norm1(latent_z)) + latent_z

        x = latent_z.reshape(b,-1)
        x = self.linear4(self.norm2(x)) 
        h = []
        for timestamp in range(action.shape[1]):
            if timestamp == 0:
                latent_x = torch.cat([x,torch.zeros_like(action[:,0])],dim=-1)
                h.append(self.decoder(latent_x))
            else:
                s = x
                s = s.unsqueeze(1)
                x = x.unsqueeze(1)
                s = self.attn4(self.norm4(s)) + s
                s = self.attn5(self.norm5_1(s),action[:,timestamp-1:timestamp])
                moments = self.linear5(s)
                b,n,c = moments.shape[0],moments.shape[1],moments.shape[2]
                moments = moments.reshape(-1,c)
                posterior = DiagonalGuassianDistribution(moments)
                s = posterior.sample()
                s = s.reshape(b,n,self.z_channels)
                s = rearrange(s,'b n c -> n b c').contiguous()
                x = rearrange(x,'b n c -> n b c').contiguous()
                x,_ = self.gru_block(s,x)
                x = x[0]
                latent_x = torch.cat([x,action[:,timestamp-1]],dim=-1)
                h.append(self.decoder(latent_x))
        return h

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

class TransformerBlock(nn.Module):
    def __init__(self,dim,n_heads,d_head,dropout=0.,context_dim=None,gated_ff=True,checkpoint=True,num_reg_fcs=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim,heads=n_heads,dim_head=d_head,dropout=dropout)
        self.ff = FeedForward(dim,dropout=dropout,glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim,context_dim=context_dim,heads=n_heads,dim_head=d_head,dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,x,context=None):
        x = self.attn1(self.norm1(x)) + x

        x = self.attn2(self.norm2(x),context=context) + x

        x = self.ff(self.norm3(x)) + x
        return x

class ActionFormer8(nn.Module):
    def __init__(self,in_channels,box_dim,flatten_dims,embed_dims,out_hdmap_channels,action_dim,depth=2,heads=8,dim_head=64,movie_len=5):
        super().__init__()
        self.in_channels = in_channels
        self.input_blocks = nn.ModuleList([TransformerBlock(in_channels,heads,dim_head,context_dim=box_dim) for i in range(depth)])
        self.linear1 = nn.Linear(flatten_dims,embed_dims)
        self.mid_blocks = nn.ModuleList([TransformerBlock(4,heads,dim_head,context_dim=action_dim) for i in range(depth)])
        self.dec_hdmap = nn.Linear(4,out_hdmap_channels)
        self.dec_dense_range_image = nn.Linear(4,out_hdmap_channels)
        self.out_blocks = nn.ModuleList([TransformerBlock(box_dim,heads,dim_head,context_dim=4) for i in range(depth)])
        self.dec_object = nn.Linear(box_dim,box_dim)
        self.movie_len = movie_len
        
        

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image):
        b,c,h,w = latent_hdmap.shape
        output = []
        output.append([latent_hdmap,latent_boxes,latent_dense_range_image])
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (h w) c').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (h w) c').contiguous()
        latent_z = torch.cat([latent_hdmap,latent_dense_range_image],dim=1)
        for i in range(self.movie_len-1):
            for block in self.input_blocks:
                latent_z = block(latent_z,latent_boxes)
            for block in self.mid_blocks:
                latent_z = block(latent_z,action)
            predict_hdmap,predict_dense_range_image = torch.chunk(latent_z,2,dim=1)
            predict_hdmap = self.dec_hdmap(predict_hdmap)
            predict_hdmap = rearrange(predict_hdmap,'b (h w) c -> b c h w',h=h,w=w).contiguous()
            predict_dense_range_image = self.dec_dense_range_image(predict_dense_range_image)
            predict_dense_range_image = rearrange(predict_dense_range_image,'b (h w) c -> b c h w',h=h,w=w).contiguous()
            object_query = latent_boxes
            for block in self.out_blocks:
                object_query = block(object_query,latent_z)
            predict_object = self.dec_object(object_query)
            output.append([predict_hdmap,predict_object,predict_dense_range_image])
            predict_hdmap = rearrange(predict_hdmap,'b c h w -> b (h w) c').contiguous()
            predict_dense_range_image = rearrange(predict_dense_range_image,'b c h w -> b (h w) c').contiguous()
            latent_z = torch.cat([predict_hdmap,predict_dense_range_image],dim=1)
            latent_boxes = predict_object
        return output

class RepresentationModel(nn.Module):
    def __init__(self,in_channels,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1
        self.module = nn.Sequential(
            nn.Linear(in_channels,in_channels),
            nn.LeakyReLU(True),
            nn.Linear(in_channels,2*self.latent_dim),
        )

    def forward(self,x):
        def sigmoid2(tensor:torch.Tensor,min_value:float) -> torch.Tensor:
            return 2*torch.sigmoid(tensor / 2) + min_value
        mu_log_sigma = self.module(x)
        mu,log_sigma = torch.split(mu_log_sigma,self.latent_dim,dim=-1)
        sigma = sigmoid2(log_sigma,self.min_std)
        return mu,sigma

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self,latent_n_channels,out_channels,epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.latent_affine = nn.Linear(latent_n_channels,2*out_channels)

    def forward(self,x,style):
        # Instance norm
        mean = x.mean(dim=(-1,-2),keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x**2,dim=(-1,-2),keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1)
        scale,bias = torch.split(style,split_size_or_sections=self.out_channels,dim=1)
        out = scale * x + bias
        return out

class ConvInstanceNorm(nn.Module):
    def __init__(self,in_channels,out_channels,latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm(latent_n_channels,out_channels)

    def forward(self,x,w):
        x = self.conv_act(x)
        return self.adaptive_norm(x,w)

class SegmentationHead(nn.Module):
    def __init__(self,in_channels,n_classes,downsample_factor):
        super().__init__()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels,n_classes,kernel_size=1,padding=0)
        )
        # self.instance_offset_head = nn.Sequential(
        #     nn.Conv2d(in_channels,n_classes,kernel_size=1,padding=0),
        # )
        # self.instance_center_head = nn.Sequential(
        #     nn.Conv2d(in_channels,1,kernel_size=1,padding=0),
        #     nn.Sigmoid(),
        # )
        self.downsample_factor = downsample_factor

    def forward(self,x):
        # output = {
        #     f'bev_sementation_{self.downsample_factor}': self.segmentation_head(x),
        #     f'bev_instance_offset_{self.downsample_factor}': self.instance_offset_head(x),
        #     f'bev_instance_center_{self.downsample_factor}': self.instance_center_head(x),
        # }
        output = self.segmentation_head(x)
        return output

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,latent_n_channels,upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm(in_channels,out_channels,latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels,out_channels,latent_n_channels)

    def forward(self,x,w):
        if self.upsample:
            x = F.interpolate(x,scale_factor=2.0,mode='bilinear',align_corners=False)
        x = self.conv1(x,w)
        return self.conv2(x,w)

class RGBHead(nn.Module):
    def __init__(self,in_channels,n_classes,downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.rgb_head = nn.Sequential(
            nn.Conv2d(in_channels,n_classes,kernel_size=1,padding=0)
        )

    def forward(self,x):
        # output = {
        #     f"rgb_{self.downsample_factor}": self.rgb_head(x)
        # }
        output = self.rgb_head(x)
        return output

class BevDecoder(nn.Module):
    def __init__(self,latent_n_channels,semantic_n_channels,constant_size=(3,3),is_segmentation=True):
        super().__init__()
        n_channels = 512
        self.constant_tensor = nn.Parameter(torch.randn((n_channels,*constant_size),dtype=torch.float32))

        # 512 x 3 x 3
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels,out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels,n_channels,latent_n_channels)
        #512 x 3 x 3

        self.middle_conv = nn.ModuleList(
            [DecoderBlock(n_channels,n_channels,latent_n_channels) for _ in range(2)]
        )

        head_module = SegmentationHead if is_segmentation else RGBHead
        self.conv1 = DecoderBlock(n_channels,256,latent_n_channels,upsample=False)

        self.conv2 = DecoderBlock(256,128,latent_n_channels,upsample=False)

        self.conv3 = DecoderBlock(128,64,latent_n_channels,upsample=False)
        self.head_1 = head_module(64,semantic_n_channels,downsample_factor=1)

    def forward(self,w):
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b,1,1,1])

        x = self.first_norm(x,w)
        x = self.first_conv(x,w)

        for module in self.middle_conv:
            x = module(x,w)
        x = self.conv1(x,w)
        x = self.conv2(x,w)
        x = self.conv3(x,w)
        output_1 = self.head_1(x)

        return output_1

class BoxDecoder(nn.Module):
    def __init__(self,in_channels,context_dim,ch,out_channels,constant_size=(70,128),heads=8,dim_head=64,depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.init_box = nn.Parameter(torch.randn((constant_size),dtype=torch.float32))

        self.input_block = nn.ModuleList([TransformerBlock(in_channels,n_heads=heads,d_head=dim_head,context_dim=context_dim) for _ in range(depth)])

        self.up1 = nn.Linear(in_channels,ch[0])

        self.mid_block = nn.ModuleList([TransformerBlock(ch[0],n_heads=heads,d_head=dim_head,context_dim=context_dim) for _ in range(depth)])

        self.up2 = nn.Linear(ch[0],ch[1])

        self.out_block = nn.ModuleList([TransformerBlock(ch[1],n_heads=heads,d_head=dim_head,context_dim=context_dim) for _ in range(depth)])

        self.out = nn.Linear(ch[1],out_channels)

    def forward(self,w):
        b = w.shape[0]
        x = self.init_box.unsqueeze(0).repeat([b,1,1])

        for block in self.input_block:
            x = block(x,w)
        x = self.up1(x)

        for block in self.mid_block:
            x = block(x,w)
        x = self.up2(x)

        for block in self.out_block:
            x = block(x,w)
        x = self.out(x)
        return x


class Policy(nn.Module):
    def __init__(self,in_channels,movie_len):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels,512),
            nn.ReLU(True),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,3*movie_len),
            nn.Tanh()
        )
    def forward(self,x):
        return self.fc(x)

class ActionFormer9(nn.Module):
    def __init__(self,hdmap_dim,flatten_dim,embedding_dim,box_dim,box_emb_dim,action_dim,hidden_state_dim,state_dim,movie_len,use_dropout=False,dropout_probability=0.0,heads=8,dim_head=64,height=16,width=32,num_boxes=70):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.hidden_state_dim = hidden_state_dim

        self.attn1 = CrossAttention(hdmap_dim,heads=heads,dim_head=dim_head)
        self.norm1 = nn.LayerNorm(hdmap_dim)
        self.attn2 = CrossAttention(hdmap_dim,context_dim=box_dim,heads=heads,dim_head=dim_head)
        self.norm2 = nn.LayerNorm(hdmap_dim)
        self.linear1 = nn.Linear(flatten_dim,hidden_state_dim)
        self.pre_gru_net = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )
        self.attn3 = CrossAttention(1,heads=heads,dim_head=dim_head)
        self.attn4 = CrossAttention(1,1,heads=heads,dim_head=dim_head)
        self.recurrent_model = nn.GRUCell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
        )

        self.posterior = RepresentationModel(
            in_channels=hidden_state_dim+action_dim+hidden_state_dim, 
            latent_dim=state_dim,
        )

        self.prior = RepresentationModel(in_channels=hidden_state_dim+action_dim,latent_dim=state_dim)

        self.hdmap_decoder = BevDecoder(latent_n_channels=hidden_state_dim,
                                        semantic_n_channels=4,
                                        constant_size=(height,width),
                                        is_segmentation=False,
                                        )
        
        self.dense_range_image_decoder = BevDecoder(latent_n_channels=hidden_state_dim,
                                                    semantic_n_channels=4,
                                                    constant_size=(height,width),
                                                    is_segmentation=False)
        
        self.box_decoder = BoxDecoder(in_channels=box_emb_dim,context_dim=1,ch=[256,512],out_channels=1024,constant_size=(70,64))

    def forward(self,latent_hdmap,latent_boxes,action,latent_dense_range_image=None,use_sample=True):
        b,c,h,w = latent_hdmap.shape
        sequence_length = action.shape[1]
        latent_hdmap = rearrange(latent_hdmap,'b c h w -> b (h w) c').contiguous()
        latent_dense_range_image = rearrange(latent_dense_range_image,'b c h w -> b (h w) c').contiguous()
        latent_z =  torch.cat([latent_hdmap,latent_dense_range_image],dim=1)
        latent_z = latent_hdmap
        latent_z = self.attn1(self.norm1(latent_z)) + latent_z
        latent_z = self.attn2(self.norm2(latent_z),latent_boxes) + latent_z
        latent_z =  latent_z.reshape(b,-1)
        latent_z = self.linear1(latent_z)
        h = []
        output = {}
        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:,0])
            else:
                action_t = action[:,t-1]
            output_t = self.observe_step(latent_z,action_t)
            sample_t = output_t['prior']['sample']
            latent_z = output_t['prior']['hidden_state']
            output = {**output,**output_t}
            h.append(latent_z)
        h = torch.stack([x for x in h],dim=1)
        h = rearrange(h,'b n c -> (b n) c').contiguous()
        dense_range_image = self.dense_range_image_decoder(h)
        hdmap_output = self.hdmap_decoder(h)
        h = h.unsqueeze(-1)
        box_output = self.box_decoder(h)
        h = h.squeeze()
        output['hdmap'] = hdmap_output
        output['boxes'] = box_output
        output['dense_range_image'] = dense_range_image
        return output
        

    def imagine_step(self,latent_z,action_t,use_sample=True):
        latent_z = latent_z.unsqueeze(-1)
        action_t = action_t.unsqueeze(-1)
        latent_z = self.attn3(latent_z) + latent_z
        latent_z = self.attn4(latent_z,action_t) + latent_z
        latent_z = latent_z.squeeze(-1)
        action_t = action_t.squeeze(-1)
        prior_mu_t,prior_sigma_t = self.prior(torch.cat([latent_z,action_t],dim=-1))
        sample_t = self.sample_from_distribution(prior_mu_t,prior_sigma_t,use_sample)
        sample_t = self.pre_gru_net(sample_t)
        h_t = self.recurrent_model(sample_t,latent_z)
        prior_mu_t,prior_sigma_t = self.prior(torch.cat([h_t,action_t],dim=-1))
        sample_t = self.sample_from_distribution(prior_mu_t,prior_sigma_t,use_sample=use_sample)
        imagine_output = {
            'hidden_state': h_t,
            'sample': sample_t,
            'mu': prior_mu_t,
            'sigma': prior_sigma_t,
            'latent_z': latent_z,
        }
        return imagine_output
        

    def observe_step(self,latent_z,action_t,use_sample=True):
        imagine_output = self.imagine_step(latent_z,action_t,use_sample)
        posterior_mu_t,posterior_sigma_t = self.posterior(torch.cat([imagine_output['hidden_state'],action_t,imagine_output['latent_z']],dim=-1))
        sample_t = self.sample_from_distribution(posterior_mu_t,posterior_sigma_t,use_sample=use_sample)
        posterior_output = {
            'hidden_state': imagine_output['hidden_state'],
            'sample': sample_t,
            'mu': posterior_mu_t,
            'sigma': posterior_sigma_t,
        }
        output = {
            'prior': imagine_output,
            'posterior': posterior_output,
        }
        return output

    @staticmethod
    def sample_from_distribution(mu,sigma,use_sample):
        sample = mu
        if use_sample:
            noise = torch.randn_like(sample)
            sample = sample + sigma * noise
        return sample

class SegmentationLoss(nn.Module):
    def __init__(self,use_top_k=False,top_k_ratio=1.0,poly_one=False,poly_one_coefficient=0.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient

    def forward(self,prediction,target):
        b,s,c,l = prediction.shape
        prediction = prediction.view(b*s,c,l)
        target = target.view(b*s,l)
        loss = F.cross_entropy(prediction,
                               target,
                               reduction='none',)
        if self.poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1-prob)
            loss = loss + loss_poly_one
        loss = loss.view(b,s,-1)
        if self.use_top_k:
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k,dim=-1)[0]
        return torch.mean(loss)

class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):
    def __init__(self,norm,ignore_index=255):
        super(SpatialRegressionLoss,self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2,but got norm = {norm}')
        
    def forward(self,prediction,target):
        # assert len(prediction.shape) == 5,"Must be 5D tensor"
        # mask = target[:,:,:1] != self.ignore_index
        # if mask.sum() == 0:
        #     return prediction.new_zeros(1)[0].float()
        loss = self.loss_fn(prediction,target,reduction='none')
        loss = torch.sum(loss,dim=-3,keepdim=True)
        return loss.mean()
    
class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""
    def __init__(self,remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self,prior_mu,prior_sigma,posterior_mu,posterior_sigma):
        posterior_var = posterior_sigma[:,1:] ** 2
        prior_var = prior_sigma[:,1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigma[:,1:])
        prior_log_sigma = torch.log(prior_sigma[:,1:])

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss
    
class KLLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=True)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss

class PipeLineActionFormer9(pl.LightningModule):
    def __init__(self,actionformer_config,global_condition_config,cond_stage_config,movie_len,action_weight,probabilistic_loss_weight,box_weight,hdmap_weight,dense_range_image_weight,scheduler_config=None,monitor='val/loss',data_type='mini',ckpt_path=None,ignore_keys=[],*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.action_former = instantiate_from_config(actionformer_config)
        self.global_condition = instantiate_from_config(global_condition_config)
        self.clip = instantiate_from_config(cond_stage_config)
        self.clip = self.clip.eval()
        self.movie_len = movie_len
        self.monitor = monitor
        self.use_scheduler = not scheduler_config is None
        self.scheduler_config = scheduler_config
        self.action_weight = action_weight
        self.probabilistic_loss_weight = probabilistic_loss_weight
        self.hdmap_weight = hdmap_weight
        self.box_weight = box_weight
        self.dense_range_image_weight = dense_range_image_weight
        self.action_loss = RegressionLoss(norm=1)
        self.probabilistic_loss = KLLoss(alpha=0.75)
        self.box_loss = RegressionLoss(norm=1)
        self.rgb_loss = SpatialRegressionLoss(norm=1)
        self.policy = Policy(in_channels=27*movie_len,movie_len=movie_len)
        for param in self.clip.parameters():
            param.requires_grad=False
        if data_type == 'mini':
            self.category_name = ['None','human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.wheelchair', 'human.pedestrian.stroller', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.construction_worker', 'animal', 'vehicle.car', 'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.trailer', 'movable_object.barrier', 'movable_object.trafficcone', 'movable_object.pushable_pullable', 'movable_object.debris', 'static_object.bicycle_rack']
        else:
            raise NotImplementedError
        self.category = {}
        for name in self.category_name:
            self.category[name] = self.clip(name).cpu().detach().numpy()
        del self.clip
        if ckpt_path:
            self.init_from_ckpt(ckpt_path,ignore_keys)

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
        x = rearrange(x,'b n h w c -> (b n) c h w')
        x = x.float()
        return x
    
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
        dense_range_image = self.get_input(batch,'dense_range_image') # (b n c h w)
        range_image = self.get_input(batch,'range_image')
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()

        boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
        for i in range(len(batch['category'])):
            for j in range(self.movie_len):
                for k in range(len(batch['category'][0][0])):
                    boxes_category[i][j][k] = self.category[batch['category'][i][j][k]]
        boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
        boxes_category = torch.tensor(boxes_category).to(self.device)
        batch['reference_image'] = ref_img
        batch['HDmap'] = hdmap
        batch['range_image'] = range_image
        batch['3Dbox'] = boxes
        batch['category'] = boxes_category
        batch['actions'] = batch['actions']
        batch['dense_range_image'] = dense_range_image
        action_origin = batch['future_trajectory']
        
        batch = {k:v.to(torch.float32) for k,v in batch.items()}
        actions = batch['actions'].clone()
        actions = actions.reshape(b,-1)
        future_trajectory = self.policy(actions)
        future_trajectory = future_trajectory.reshape(future_trajectory.shape[:-1]+(self.movie_len,3))
        out = self.global_condition.get_conditions(batch)
        x = out['ref_image'].to(torch.float32)
        c = {}
        c['hdmap'] = out['hdmap'].to(torch.float32)
        c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        c['actions'] = out['actions']
        c['dense_range_image'] = out['dense_range_image']
        loss,loss_dict = self(x,c,batch,action_origin,future_trajectory)
        return loss,loss_dict
    
    def forward(self,x,c,batch,action_origin,future_trajectory):
        return self.p_losses(x,c,batch,action_origin,future_trajectory)
    
    def apply_model(self,cond):
        boxes_emb = cond['boxes_emb']
        dense_range_image = cond['dense_range_image']
        b = boxes_emb.shape[0] // self.movie_len
        boxes_emb = boxes_emb.reshape((b,self.movie_len)+boxes_emb.shape[1:])
        boxes_emb = boxes_emb[:,0]
        hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
        dense_range_image = dense_range_image.reshape((b,self.movie_len)+dense_range_image.shape[1:])[:,0]
        actions = cond['actions']
        output = self.action_former(hdmap,boxes_emb,actions,dense_range_image)
        return output

    @torch.no_grad() 
    def log_video(self,batch,only_inputs=False,**kwargs):
        log = dict()
        b = batch['reference_image'].shape[0]
        log['inputs'] = rearrange(batch['HDmap'].clone(),'b n h w c -> b n c h w')
        log['dense_inputs'] = rearrange(batch['dense_range_image'],'b n h w c -> b n c h w')
        range_image = self.get_input(batch,'range_image')
        #print(batch['HDmap'].shape)
        ref_img = self.get_input(batch,'reference_image') # (b n c h w)
        hdmap = self.get_input(batch,'HDmap') # (b n c h w)
        dense_range_image = self.get_input(batch,'dense_range_image') # (b n c h w)
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()
        boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
        for i in range(len(batch['category'])):
            for j in range(self.movie_len):
                for k in range(len(batch['category'][0][0])):
                    boxes_category[i][j][k] = self.category[batch['category'][i][j][k]]
        boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
        boxes_category = torch.tensor(boxes_category).to(self.device)
        batch['reference_image'] = ref_img
        batch['HDmap'] = hdmap
        batch['3Dbox'] = boxes
        batch['range_image'] = range_image
        batch['category'] = boxes_category
        batch['actions'] = batch['actions']
        action_origin = batch['future_trajectory']
        batch['dense_range_image'] = dense_range_image
        batch = {k:v.to(torch.float32) for k,v in batch.items()}
        actions = batch['actions'].clone()
        actions = actions.reshape(b,-1)
        future_trajectory = self.policy(actions)
        future_trajectory = future_trajectory.reshape(future_trajectory.shape[:-1]+(self.movie_len,3))
        with torch.no_grad():
            out = self.global_condition.get_conditions(batch)
        x = out['ref_image'].to(torch.float32)
        c = {}
        c['hdmap'] = out['hdmap'].to(torch.float32)
        c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        c['actions'] = out['actions']
        c['dense_range_image'] = out['dense_range_image']
        c['trajectory'] = future_trajectory
        output = self.apply_model(c)
        latent_hdmap = self.global_condition.decode_first_stage_interface('reference_image',output['hdmap'])
        latent_dense_range_image = self.global_condition.decode_first_stage_interface('reference_image',output['dense_range_image'])
        log['reconstruction'] = latent_hdmap.reshape((b,self.movie_len)+latent_hdmap.shape[1:])
        log['lidar_reconstruction'] = latent_dense_range_image.reshape((b,self.movie_len)+latent_hdmap.shape[1:])
        return log

    def p_losses(self,x_start,cond,batch,action_origin,future_trajectory):
        output = self.apply_model(cond)
        # trajectory = self.action_weight * self.action_loss(future_trajectory,action_origin).mean()
        # losses['trajectory'] = trajectory
        probabilistic_loss = self.probabilistic_loss_weight * self.probabilistic_loss(output['prior'],output['posterior'])
        # losses['kl_loss'] = probabilistic_loss
        boxes_loss = self.box_weight * self.box_loss(
            prediction=output['boxes'],
            target=cond['boxes_emb'],
        ).mean()
        # loss['boxes_loss'] = boxes_loss
        hdmap_loss = self.hdmap_weight * self.rgb_loss(output['hdmap'],
                                   cond['hdmap']).mean()

        dense_range_image_loss = self.dense_range_image_weight * self.rgb_loss(output['dense_range_image'],
                                                                          cond['dense_range_image']).mean()
        # loss['hdmap_loss'] = hdmap_loss
        
        prefix = 'train' if self.training else 'val'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_simple': hdmap_loss + boxes_loss + probabilistic_loss + dense_range_image_loss})
        # loss_dict.update({f'{prefix}/loss_simple': loss_hdmap + loss_boxes + loss_dense})
        loss_dict.update({f'{prefix}/loss_hdmap':hdmap_loss})
        loss_dict.update({f'{prefix}/loss_boxes':boxes_loss})
        loss_dict.update({f'{prefix}/loss_dsr':dense_range_image_loss})
        loss_dict.update({f'{prefix}/probabilistic_loss':probabilistic_loss})
        # loss_dict.update({f'{prefix}/trajectory_loss':trajectory})
        # loss = trajectory + boxes_loss + hdmap_loss + probabilistic_loss
        loss = boxes_loss + hdmap_loss + probabilistic_loss + dense_range_image_loss
        # loss_dict.update({f'{prefix}/loss_dense':loss_dense})
        return loss,loss_dict       



# class ActionFormer7(nn.Module):
#     def __init__(
#             self,
#             encoder_config,
#             latent_hdmap_dim,
#             num_classes=3,#map classes
#             embed_dim=64,
#             num_heads=8,
#             dim_head=64,
#             action_dims=128,
#             depth=2,
#             num_vec=50,
#             num_pts_per_vec=8,
#             num_pts_per_gt_vec=8,
#             loss_pts_config=None,
#             loss_dir_config=None,
#             loss_cls_config=None,
#             loss_bbox_config=None,
#             loss_iou_config=None,
#             transformer_config=None,
#             query_embed_type='all_pts'
#             ):
#         super().__init__()
#         self.encoder = instantiate_from_config(encoder_config)
#         self.hdmap_linear = nn.Linear(latent_hdmap_dim,embed_dim)
#         self.rotation_linear = nn.Linear(4,action_dims)
#         self.translation_linear = nn.Linear(3,action_dims)
#         self.velocity_linear = nn.Linear(3,action_dims)
#         self.embed_dim = embed_dim
#         self.transformer_blocks = nn.ModuleList(
#             TransformerBlock(embed_dim,num_heads,dim_head,context_dim=action_dims) for _ in range(depth)
#         )
#         self.loss_pts = instantiate_from_config(loss_pts_config)
#         self.loss_dir = instantiate_from_config(loss_dir_config)
#         self.loss_cls = instantiate_from_config(loss_cls_config)
#         self.loss_bbox = instantiate_from_config(loss_bbox_config)
#         self.loss_iou = instantiate_from_config(loss_iou_config)
#         self.transfromer = instantiate_from_config(transformer_config)
#         num_query = num_vec * num_pts_per_vec
#         self.num_query = num_query
#         self.num_vec = num_vec
#         self.num_pts_per_vec = num_pts_per_vec
#         self.num_pts_per_gt_vec = num_pts_per_gt_vec
#         if self.loss_cls.use_sigmoid:
#             self.cls_out_channels = num_classes
#         else:
#             self.cls_out_channels = num_classes + 1
#         self.query_embed_type = query_embed_type
#         self._init_layers()

#     def _init_layers(self):
#         """Initialize classification branch and regression branch of head."""
#         cls_branch = []
#         for _ in range(self.num_reg_fcs):
#             cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#             cls_branch.append(nn.LayerNorm(self.embed_dims))
#             cls_branch.append(nn.ReLU(inplace=True))
#         cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
#         fc_cls = nn.Sequential(*cls_branch)

#         reg_branch = []
#         for _ in range(self.num_reg_fcs):
#             reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#             reg_branch.append(nn.ReLU())
#         reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
#         reg_branch = nn.Sequential(*reg_branch)
        
#         def _get_clones(module, N):
#             return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#         # last reg_branch is used to generate proposal from
#         # encode feature map when as_two_stage is True.
#         num_pred = self.transformer.decoder.num_layers
        
#         self.cls_branches = _get_clones(fc_cls, num_pred)
#         self.reg_branches = _get_clones(reg_branch, num_pred)

#         if self.query_embed_type == 'instance_pts':
#             self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dim * 2)
#             self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dim * 2)

#     def forward(self,bev_hdmap,actions):
#         bev_hdmap = self.encoder(bev_hdmap)
#         bev_hdmap = self.hdmap_linear(bev_hdmap)
#         rotation = actions[:4]
#         velocity = actions[4:7]
#         translation = actions[7:10]
#         rotation = self.rotation_linear(rotation)
#         velocity = self.velocity_linear(velocity)
#         translation = self.translation_linear(translation)

#         actions = torch.cat([rotation,velocity,translation],dim=1)
        
#         for d in range(len(self.transformer_blocks)):
#             bev_hdmap = self.transformer_blocks[d](bev_hdmap,context=actions)
        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/actionformer9_train_mini.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    # actionformer = instantiate_from_config(cfg['model']['params']['actionformer_config'])
    
    actionformer = instantiate_from_config(cfg['model'])
    latent_hdmap = torch.randn((2,5,128,256,3))
    dense_range_image = torch.randn((2,5,128,256,3))
    latent_boxes = torch.randn((2,5,70,16))
    reference_image = torch.randn((2,5,128,256,3))
    box_category = [[["None" for k in range(70)] for i in range(5)] for j in range(2)]
    # latent_bev = torch.randn(2,4,64,64)
    action = torch.randn(2,5,27)
    future_trajectory = torch.randn(2,5,3)
    batch = {
        'HDmap': latent_hdmap,
        '3Dbox': latent_boxes,
        'actions': action,
        'reference_image': reference_image,
        'category':box_category,
        'future_trajectory':future_trajectory,
        'dense_range_image':dense_range_image,
    }
    actionformer.log_video(batch)
