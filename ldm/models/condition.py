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
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img,exists,default,ismap,isimage,mean_flat, count_params, instantiate_from_config,to_cpu
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGuassianDistribution
from ldm.models.autoencoder import AutoencoderKL,VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.attention import PositionalEncoder
from ldm.modules.diffusionmodules.util import extract_into_tensor
import omegaconf
import time
import concurrent.futures
import threading

class GlobalCondition(pl.LightningModule):
    def __init__(self,
                 image_config,
                 lidar_config,
                 box_config,
                 split_input_params=None,
                 scale_factor=1.,
                 learning_rate=None,
                 action_encoder_config=None,
                 *args,**kwargs,
                 ):
        super().__init__()
        self.image_model = instantiate_from_config(image_config)
        self.lidar_model = instantiate_from_config(lidar_config)
        self.configure_learning_rate(learning_rate)
        if not self.image_model.trainable:
            self.image_model = self.image_model.eval()
            for param in self.image_model.parameters():
                param.requires_grad = False
        else:
            # self.image_model.optimizer,_ = self.image_model.configure_optimizers()
            for param in self.image_model.encoder.parameters():
                param.requires_grad = False
        
        if not self.lidar_model.trainable:
            self.lidar_model = self.lidar_model.eval()
            for param in self.lidar_model.parameters():
                param.requires_grad = False
        else:
            for param in self.image_model.encoder.parameters():
                param.requires_grad = False
            # self.lidar_model.optimizer,_ = self.lidar_model.configure_optimizers()
            # for name,param in self.lidar_model.named_parameters():
            #     if 'encoder' in name or ('quant_conv' in name and not 'post_quant_conv' in name):
            #         param.requires_grad=False
        self.box_encoder = instantiate_from_config(box_config)
        if split_input_params:
            self.split_input_params = split_input_params
        self.scale_factor = scale_factor
        if not action_encoder_config is None:
            self.action_encoder = instantiate_from_config(action_encoder_config)
        else:
            self.action_encoder = None

    def get_first_stage_encoding(self,encoder_posterior):
        if isinstance(encoder_posterior,DiagonalGuassianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior,torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}'")
        return z
    
    def get_parameters(self,training_strategy='full'):
        param = list()
        if training_strategy == 'full':
            if self.lidar_model.trainable:
                param = param + list(self.lidar_model.decoder.parameters())
            if self.image_model.trainable:
                param = param + list(self.image_model.decoder.parameters())
            if self.box_encoder.trainable:
                param = param + list(self.box_encoder.parameters())
            if not self.action_encoder is None and self.action_encoder.trainable:
                print("add action_encoder parameters")
                param = param + list(self.action_encoder.parameters())
        elif 'multimodal' in training_strategy:
            #TODO: trainable False -> True
            param = param + list(self.lidar_model.decoder.parameters())
            param = param + list(self.image_model.decoder.parameters())
        else:
            raise NotImplementedError
        return param

    def encode_first_stage(self,x,encoder,return_temp_output=False):
        if hasattr(self,'split_input_params'):
            if self.split_input_params['patch_distributed_vq']:
                ks = self.split_input_params["ks_enc"] #eg. (128,128)
                stride = self.split_input_params["stride_enc"] # eg. (64,64)
                df = self.split_input_params["vqf"]
                # self.split_input_params["original_image_size"] = x.shape[-2:]
                bs,nc,h,w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0],h),min(ks[1],w))
                    print("reducing Kernel")
                
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0],h) ,min(stride[1],w))
                    print("reducing stride")
                
                fold,unfold,normalization,weighting = self.get_fold_unfold(x,ks,stride,df=df)
                z = unfold(x) # (bn,nc,prod(**ks),L)
                #reshape to img shape
                z = z.view((z.shape[0],-1,ks[0],ks[1],z.shape[-1])) #(bn,nc,ks[0],ks[1],L)
                output_list = [self.get_first_stage_encoding(encoder.encode(z[:,:,:,:,i])) for i in range(z.shape[-1])]
                o = torch.stack(output_list,axis=-1)
                o = o * weighting
                
                # Reverse reshape to img shape
                o = o.view((o.shape[0],-1,o.shape[-1])) # (bn,nc*ks[0]*ks[1],L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            else:
                return encoder.encode(x,return_temp_output)
        else:
            return encoder.encode(x,return_temp_output)
        
    def get_fold_unfold(self,x,kernel_size,stride,uf=1,df=1):# todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs,nc,h,w = x.shape

        #number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1
        
        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size,dilation=1,padding=0,stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:],**fold_params)

            weighting = self.get_weighting(kernel_size[0],kernel_size[1],Ly,Lx,x.device).to(x.dtype)
            normalization = fold(weighting).view(1,1,h,w) #normalizes the overlap 
            weighting = weighting.view((1,1,kernel_size[0],kernel_size[1],Ly*Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size,dilation=1,padding=0,stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            #TODO:check kernel size
            fold_params2 = dict(kernel_size=(kernel_size[0]*uf,kernel_size[1]*uf),
                                dilation=1,padding=0,
                                stride=(stride[0]*uf,stride[1]*uf))
            fold = torch.nn.Fold(output_size=(x.shape[2]*uf,x.shape[3]*uf),**fold_params2)

            weighting = self.get_weighting(kernel_size[0]*uf,kernel_size[1]*uf,Ly,Lx,x.device).to(x.dtype)
            normalization = fold(weighting).view(1,1,h*uf,w*uf)
            weighting = weighting.view((1,1,kernel_size[0]*uf,kernel_size[1]*uf,Ly*Lx))
        elif df>1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size,dilation=1,padding=0,stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            
            #TODO:check kernel size
            fold_params2 = dict(kernel_size=(kernel_size[0] // df,kernel_size[1] // df),
                                dilation=1,padding=0,
                                stride=(stride[0] // df,stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df,x.shape[3] // df),**fold_params2)
            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1,1,h//df,w//df)
            weighting = weighting.view((1,1,kernel_size[0]//df,kernel_size[1]//df,Ly*Lx))
        else:
            raise NotImplementedError
        return fold,unfold,normalization,weighting
    
    def get_weighting(self,h,w,Ly,Lx,device):
        weighting = self.delta_border(h,w)
        weighting = torch.clip(weighting,self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"],)
        weighting = weighting.view(1,h*w,1).repeat(1,1,Ly*Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly,Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            L_weighting = L_weighting.view(1,1,Ly*Lx).to(device)
            weighting = weighting * L_weighting
        return weighting
    
    def delta_border(self,h,w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h,w]).view(1,1,2)
        arr = self.meshgrid(h,w) / lower_right_corner
        dist_left_up = torch.min(arr,dim=-1,keepdims=True)[0]
        dist_right_down = torch.min(1-arr,dim=-1,keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up,dist_right_down],dim=-1),dim=-1)[0]
        return edge_dist
    
    def meshgrid(self,h,w):
        y = torch.arange(0,h).view(h,1,1).repeat(1,w,1)
        x = torch.arange(0,w).view(1,w,1).repeat(h,1,1)
        arr = torch.cat([y,x],dim=-1)
        return arr
    
    def decode_first_stage_interface(self,model_name,z,predict_cids=False,force_not_quantize=False,calc_decoder_loss=False):
        if model_name == "reference_image":
            return self.decode_first_stage(self.image_model,z,predict_cids,force_not_quantize,calc_decoder_loss)
        elif model_name == "lidar":
            return self.decode_first_stage(self.lidar_model,z,predict_cids,force_not_quantize,calc_decoder_loss)
    
    def decode_first_stage(self,model,z,predict_cids=False,force_not_quantize=False,calc_decoder_loss=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1. / self.scale_factor * z

        if hasattr(self,"split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks_enc"]
                stride = self.split_input_params["stride_enc"]
                uf = self.split_input_params["vqf"]
                bs,nc,h,w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")
                
                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            
                output_list = [model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]
                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                return model.decode(z)
        else:
            if isinstance(model,VQModelInterface):
                return model.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return model.decode(z,return_temp_output=calc_decoder_loss)
            
    def get_conditions(self,batch,calc_decoder_loss=False):
        condition_keys = batch.keys()
        out = {}
        ref_image = batch['reference_image']
        z = self.encode_first_stage(ref_image,self.image_model,calc_decoder_loss)
        if calc_decoder_loss:
            z,cam_enc = z
            out['cam_enc'] = cam_enc
        ref_image = self.get_first_stage_encoding(z)
        out['ref_image'] = ref_image
        if 'HDmap' in condition_keys:
            hdmap = batch['HDmap']
            hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.image_model))
            out['hdmap'] = hdmap
        if 'range_image' in condition_keys:
            range_image = batch['range_image']
            lidar_z = self.encode_first_stage(range_image,self.lidar_model,calc_decoder_loss)
            if calc_decoder_loss:
                lidar_z,lidar_enc = lidar_z
                out['lidar_enc'] = lidar_enc
            range_image = self.get_first_stage_encoding(lidar_z)
            out['range_image'] = range_image
        if 'dense_range_image' in condition_keys:
            dense_range_image = batch['dense_range_image']
            dense_range_image = self.get_first_stage_encoding(self.encode_first_stage(dense_range_image,self.lidar_model))
            out['dense_range_image'] = dense_range_image
        if '3Dbox' in condition_keys:
            boxes = batch['3Dbox']
            box_category = batch['category']
            boxes_emb = self.box_encoder(boxes,box_category)
            out['boxes_emb'] = boxes_emb
        if 'text' in condition_keys:
            text_emb = batch['text']
            out['text_emb'] = text_emb
        if 'actions' in condition_keys:
            actions = batch['actions']
            if not self.action_encoder is None:
                actions_embed = self.action_encoder(actions)
            else:
                actions_embed = actions
            out['actions'] = actions_embed
        if 'bev_images' in condition_keys:
            bev_image = batch['bev_images']
            bev_image = self.get_first_stage_encoding(self.encode_first_stage(bev_image,self.image_model))
            out['bev_images'] = bev_image
        return out
    
    def calc_single_similarity(self,cam_enc,lidar_enc,use_similar):
        eps = 1e-7
        data = torch.tensor([0.],dtype=torch.float32,device=cam_enc[0].device)
        if use_similar == 'JS':
            for i in range(len(cam_enc)):
                min_value = cam_enc[i].min()
                max_value = cam_enc[i].max()
                cam_enc[i] = (cam_enc[i] - min_value) / (max_value - min_value)
                _,_,h,w = cam_enc[i].shape
                l = h * w
                cam_enc[i] = rearrange(cam_enc[i],'b c h w -> (b h w) c')
                min_value = lidar_enc[i].min()
                max_value = lidar_enc[i].max()
                lidar_enc[i] = (lidar_enc[i] - min_value) / (max_value - min_value)
                JS =  torch.tensor([0.],device=cam_enc[i].device)
                for j in range(cam_enc[i].shape[1]):
                    xx = torch.histc(cam_enc[i][:,j],bins=256)
                    lidar_xx = torch.histc(lidar_enc[i][:,j],bins=256)
                    lidar_xx = lidar_xx / l + eps
                    xx = xx / l + eps
                    m = (xx + lidar_xx) * 0.5
                    kl_pm = torch.sum((torch.kl_div(xx,m)))
                    kl_qm = torch.sum((torch.kl_div(lidar_xx,m)))
                    js = 0.5 * (kl_pm + kl_qm)
                    JS += js
                data += JS
        return data
        

    def calc_similarity_loss(self,cam_enc,cam_dec,lidar_enc,lidar_dec,use_similar='JS'):
        if use_similar == "JS":
            gt = self.calc_single_similarity(cam_enc,lidar_enc,use_similar)
            rec = self.calc_single_similarity(cam_dec,lidar_dec,use_similar)
            similarity = (gt - rec).mean()
            return similarity

    def get_losses(self,batch):
        losses = 0
        log_dict = {}
        if self.image_model.trainable:
            ref_ae_loss,ref_log_dict_ae = self.image_model.get_losses(batch['reference_image'],0,optimizer_idx=0)
            ref_log_dict_ae = {k+"_ref":v for k,v in ref_log_dict_ae.items()}
            hdmap_ae_loss,hdmap_log_dict_ae = self.image_model.get_losses(batch['HDmap'],0,optimizer_idx=0)
            hdmap_log_dict_ae = {k+"_hdmap":v for k,v in hdmap_log_dict_ae.items()}
            ref_disc_loss,ref_log_dict_disc = self.image_model.get_losses(batch['reference_image'],0,optimizer_idx=1)
            ref_log_dict_disc = {k+"_ref":v for k,v in ref_log_dict_disc.items()}
            hdmap_disc_loss,hdmap_log_dict_disc = self.image_model.get_losses(batch['HDmap'],0,optimizer_idx=1)
            hdmap_log_dict_disc = {k+"_hdmap":v for k,v in hdmap_log_dict_disc.items()}
            losses = ref_ae_loss + hdmap_ae_loss + ref_disc_loss,hdmap_disc_loss
            log_dict = {k:v if not k in log_dict.keys() else v+log_dict[k] for k,v in ref_log_dict_ae.items()}
            log_dict = {k:v if not k in log_dict.keys() else v+log_dict[k] for k,v in hdmap_log_dict_ae.items()}
            log_dict = {k:v if not k in log_dict.keys() else v+log_dict[k] for k,v in ref_log_dict_disc.items()}
            log_dict = {k:v if not k in log_dict.keys() else v+log_dict[k] for k,v in hdmap_log_dict_disc.items()}

        if self.lidar_model.trainable:
            lidar_ae_loss,lidar_log_dict_ae = self.lidar_model.get_losses(batch['range_image'],0,optimizer_idx=0,target=batch['range_image'])
            lidar_log_dict_ae = {k+"_lidar":v for k,v in lidar_log_dict_ae.items()}
            depth_ae_loss,depth_log_dict_ae = self.lidar_model.get_losses(batch['dense_range_image'],0,optimizer_idx=0)
            depth_log_dict_ae = {k+"_depth":v for k,v in depth_log_dict_ae.items()}
            lidar_disc_loss,lidar_log_dict_disc = self.lidar_model.get_losses(batch['range_image'],0,optimizer_idx=1,target=batch['range_image'])
            lidar_log_dict_disc = {k+"_lidar":v for k,v in lidar_log_dict_disc.items()}
            depth_disc_loss,depth_log_dict_disc = self.lidar_model.get_losses(batch['dense_range_image'],0,optimizer_idx=1)
            depth_log_dict_disc = {k+"_depth":v for k,v in depth_log_dict_disc.items()}
            losses += lidar_ae_loss+depth_ae_loss+lidar_disc_loss+depth_disc_loss
            for k,v in lidar_log_dict_ae.items():
                if not k in log_dict.keys():
                    log_dict[k] = v
                else:
                    log_dict[k] += v
            for k,v in depth_log_dict_ae.items():
                if not k in log_dict.keys():
                    log_dict[k] = v
                else:
                    log_dict[k] += v
            for k,v in lidar_log_dict_disc.items():
                if not k in log_dict.keys():
                    log_dict[k] = v
                else:
                    log_dict[k] += v
            for k,v in depth_log_dict_disc.items():
                if not k in log_dict.keys():
                    log_dict[k] = v
                else:
                    log_dict[k] += v
        return losses,log_dict

    def get_scale_factor(self,batch):
        print("### USING STD-RESCALING ###")
        ref_img = batch['reference_image']
        ref_img = rearrange(ref_img,'b n h w c -> (b n) c h w')
        ref_img = ref_img.to(memory_format=torch.contiguous_format).float()
        z = self.get_first_stage_encoding(self.encode_first_stage(ref_img,self.image_model)).detach()
        del self.scale_factor
        self.register_buffer('scale_factor',1. / z.flatten().std())
        print(f"setting self.scale_factor to {self.scale_factor}")
        print("### USING STD-RESCALING ###")
        return

    def get_parameters_name(self):
        param_names = []
        for name,param in self.named_parameters():
            model_name = name.split('.')[0]
            if model_name == "image_model":
                if self.image_model.trainable:
                    param_names.append("global_condition."+name)
            elif model_name == "lidar_model":
                if self.lidar_model.trainable:
                    param_names.append("global_condition."+name)
            elif model_name == "box_encoder":
                if self.box_encoder.trainable:
                    param_names.append("global_condition."+name)
        return param_names
    
    def configure_learning_rate(self,learning_rate):
        if self.image_model.trainable:
            self.image_model.learning_rate=learning_rate
        if self.lidar_model.trainable:
            self.lidar_model.learning_rate = learning_rate
        


