import sys
sys.path.append('.')

import random
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ldm.modules.diffusionmodules.util import fourier_filter
from ldm.util import append_dims, instantiate_from_config
from .denoiser import Denoiser
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from ldm.modules.losses.discriminator import hinge_d_loss,adopt_weight,new_d_loss,vanilla_new_d_loss,vanilla_d_loss
from ldm.modules.diffusionmodules.util import make_beta_schedule,extract_into_tensor
from functools import partial
from ldm.util import exists,default

class StandardDiffusionLoss(nn.Module):
    def __init__(
            self,
            sigma_sampler_config: dict,
            loss_weighting_config: dict,
            loss_type: str = "l2",
            use_additional_loss: bool = False,
            offset_noise_level: float = 0.0,
            additional_loss_weight: float = 0.0,
            movie_len: int = 25,
            replace_cond_frames: bool = False,
            cond_frames_choices: Union[List, None] = None,
            img_size: tuple = (128,256),
            depth_config=None,
            w_similarity=1.,
            discriminator_config = None,
            disc_factor=1.,
            disc_start=0.,
            disc_weight=1.0,
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"]
        self.loss_type = loss_type
        self.use_additional_loss = use_additional_loss

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.offset_noise_level = offset_noise_level
        self.additional_loss_weight = additional_loss_weight
        self.movie_len = movie_len
        self.replace_cond_frames = replace_cond_frames
        self.cond_frames_choices = cond_frames_choices
        if depth_config is not None:
            self.depth_estimator = instantiate_from_config(depth_config)
            self.img_size = img_size
            self.theta_up = np.pi / 12 
            self.theta_down = -np.pi / 6
            self.theta_res = (self.theta_up - self.theta_down) / self.img_size[0]
            self.phi_res = (np.pi / 3) / self.img_size[1]
        else:
            self.depth_estimator = None
        if discriminator_config is not None:
            self.discriminator = instantiate_from_config(discriminator_config)
            self.disc_loss = vanilla_new_d_loss
            self.disc_factor = disc_factor
            self.discriminator_iter_start = disc_start
            self.disc_weight = disc_weight
        self.w_similarity = w_similarity
    def get_noised_input(
            self,
            sigmas_bc: torch.Tensor,
            noise: torch.Tensor,
            input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input
    
    def forward(
            self,
            network:nn.Module,
            denoiser: Denoiser,
            cond:dict,
            x:torch.Tensor,
            range_image:torch.Tensor,
            actionformer:nn.Module,
            global_step:int,
            calc_dec_loss:bool = False,
            use_similar: str = "JS",
            optimizer_idx: int=None,
            last_layer=None,
    ):
        return self._forward(network,denoiser,cond,x,range_image,actionformer,global_step,calc_dec_loss,use_similar,optimizer_idx,last_layer)
    
    def calc_kl_distance(self,x,range_image):
        mu1,var1 = torch.mean(x,dim=0),torch.var(x,dim=0)
        mu2,var2 = torch.mean(range_image,dim=0),torch.var(range_image,dim=0)
        var1 = torch.clamp(var1,min=1e-10)
        var2 = torch.clamp(var2,min=1e-10)
        return torch.log(var2/var1) + torch.div(torch.add(torch.square(var1),torch.square(mu1-mu2)),2*torch.square(var2)) - 0.5

    def calculate_adaptive_weight(self,loss,g_loss,last_layer=None):
        if last_layer is not None:
            loss_grads = torch.autograd.grad(loss,last_layer,retain_graph=True)[0]
            g_grad = torch.autograd.grad(g_loss,last_layer,retain_graph=True)[0]
        else:
            raise not NotImplementedError
        d_weight = torch.norm(loss_grads) / (torch.norm(g_grad) + 1e-4)
        d_weight = torch.clamp(d_weight,0.0,1e-4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def _forward(
            self,
            network:nn.Module,
            denoiser:Denoiser,
            cond:Dict,
            x:torch.Tensor,
            range_image:torch.Tensor,
            actionformer:nn.Module,
            global_step: int,
            calc_dec_loss:bool = False,
            use_similar: str = "JS",
            optimizer_idx: int=None,
            last_layer=None,
    ):
        sigmas = self.sigma_sampler(x.shape[0]).to(x)
        cond_mask = torch.zeros_like(sigmas)
        if self.replace_cond_frames:
            cond_mask = rearrange(cond_mask,"(b t) -> b t",t=self.movie_len)
            for each_cond_mask in cond_mask:
                assert len(self.cond_frames_choices[-1]) < self.movie_len
                weights = [2**n for n in range(len(self.cond_frames_choices))]
                cond_indices = random.choices(self.cond_frames_choices,weights=weights,k=1)[0]
                if cond_indices:
                    each_cond_mask[cond_indices] = 1
            cond_mask = rearrange(cond_mask,"b t -> (b t)")
        if calc_dec_loss:
            gt_kl_distance = self.calc_kl_distance(x,range_image)
        else:
            gt_kl_distance = None
        noise = torch.randn_like(x)
        if self.offset_noise_level > 0.0:
            offset_shape = (x.shape[0],x.shape[1])
            rand_init = torch.randn(offset_shape,device=x.device)
            noise = noise + self.offset_noise_level * append_dims(rand_init,x.ndim)
        if self.replace_cond_frames:
            sigmas_bc = append_dims((1-cond_mask)*sigmas,x.ndim)
        else:
            sigmas_bc = append_dims(sigmas,x.ndim)
        noised_x = self.get_noised_input(sigmas_bc,noise,x)
        if not range_image is None:
            noised_range_image = self.get_noised_input(sigmas_bc,noise,range_image)
        else:
            noised_range_image = None
        model_output = denoiser(network,noised_x,noised_range_image,sigmas,cond,self.movie_len,actionformer,cond_mask)
        if range_image is None:
            x_rec = model_output
        else:
            x_rec,range_image_rec = torch.chunk(model_output,2,0)
        if range_image is None:
            sigmas = sigmas
        else:
            sigmas = torch.cat([sigmas,sigmas],dim=0)
        w = append_dims(self.loss_weighting(sigmas),x.ndim)
        if self.replace_cond_frames:
            predict_x = x_rec * append_dims(1 - cond_mask,x.ndim) + x * append_dims(cond_mask,x.ndim)
            if not range_image is None:
                predict_range = range_image_rec * append_dims(1 - cond_mask,x.ndim) + range_image * append_dims(cond_mask,x.ndim)
                predict = torch.cat([predict_x,predict_range],dim=0)
            else:
                predict = predict_x
        else:
            predict = model_output
        if not range_image is None:
            input = torch.cat([x,range_image],dim=0)
        else:
            input = x
        if optimizer_idx is None:
            return self.get_loss(predict,input,w,cond,global_step,calc_dec_loss,use_similar,multimodal=(not range_image is None),gt_kl_distance=gt_kl_distance)
        else:
            if optimizer_idx == 0:
                loss,camera_loss,range_image_loss = self.get_loss(predict,input,w,cond,global_step,calc_dec_loss,use_similar,multimodal=(not range_image is None),gt_kl_distance=gt_kl_distance)
                fake_x,fake_range_image = torch.chunk(predict,2,0)
                fake_inputs = torch.cat([fake_x,fake_range_image],dim=1)
                logits_fake = self.discriminator(fake_inputs)
                g_loss = -torch.mean(logits_fake)
                disc_factor = adopt_weight(self.disc_factor,global_step,threshold=self.discriminator_iter_start)
                d_weight = self.calculate_adaptive_weight(loss,g_loss,last_layer)
                loss = loss + disc_factor * g_loss * d_weight
                return loss,camera_loss,range_image_loss,g_loss
            else:
                if hasattr(self,'discriminator'):
                    gt_x,gt_range_image = torch.chunk(input,2,0)
                    inputs = torch.cat([gt_x,gt_range_image],dim=1)
                    logits_real = self.discriminator(inputs)
                    fake_x,fake_range_image = torch.chunk(predict,2,0)
                    fake_inputs = torch.cat([fake_x,fake_range_image],dim=1)
                    logits_fake = self.discriminator(fake_inputs)
                    if self.disc_loss is new_d_loss or self.disc_loss is vanilla_new_d_loss:
                        gt_x_fake_r = torch.cat([gt_x,fake_range_image],dim=1)
                        fake_x_gt_r = torch.cat([fake_x,gt_range_image],dim=1)
                        logits_fake_1 = self.discriminator(gt_x_fake_r)
                        logits_fake_2 = self.discriminator(fake_x_gt_r)
                        logits_fake = torch.cat([logits_fake,logits_fake_1,logits_fake_2],dim=0)
                    disc_factor = adopt_weight(self.disc_factor,global_step,threshold=self.discriminator_iter_start)
                    d_loss = disc_factor * self.disc_loss(logits_real,logits_fake)

                    d_loss = torch.mean(d_loss)
                    logits_fake = torch.mean(logits_fake)
                    logits_real = torch.mean(logits_real)
                    return d_loss,logits_real,logits_fake
                else:
                    assert 0,"discriminator is None"

    
    def calc_single_similarity(self,cam_enc,lidar_enc,use_similar):
        eps = 1e-7
        data = []
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
                JS =  []
                for j in range(cam_enc[i].shape[1]):
                    xx = torch.histc(cam_enc[i][:,j],bins=256)
                    lidar_xx = torch.histc(lidar_enc[i][:,j],bins=256)
                    lidar_xx = lidar_xx / l + eps
                    xx = xx / l + eps
                    m = (xx + lidar_xx) * 0.5
                    kl_pm = torch.sum((torch.kl_div(xx,m)))
                    kl_qm = torch.sum((torch.kl_div(lidar_xx,m)))
                    js = 0.5 * (kl_pm + kl_qm)
                    JS.append(js)
                JS = torch.tensor(JS,dtype=torch.float32,device=cam_enc[0].device)
                JS = torch.mean(JS)
                data.append(JS)
        data = torch.tensor(data,dtype=torch.float32,device=cam_enc[0].device)
        return torch.mean(data)
        

    def calc_similarity_loss(self,cam_enc,cam_dec,lidar_enc,lidar_dec,use_similar='JS'):
        if use_similar == "JS":
            gt = self.calc_single_similarity(cam_enc,lidar_enc,use_similar)
            rec = self.calc_single_similarity(cam_dec,lidar_dec,use_similar)
            similarity = (gt - rec).mean()
            return similarity

    def get_loss(self,predict,target,w,cond,global_step,calc_dec_loss=False,use_similar="JS",multimodal=True,gt_kl_distance=None):
        if calc_dec_loss:
            cam_enc,cam_dec,lidar_enc,lidar_dec = cond['cam_enc'],cond['cam_dec'],cond['lidar_enc'],cond['lidar_dec']
            similarity = self.calc_similarity_loss(cam_enc,cam_dec,lidar_enc,lidar_dec,use_similar)

        if self.loss_type == "l2":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.movie_len)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.movie_len)
                bs = target.shape[0] // self.movie_len
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])) ** 2
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=2)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.movie_len - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf) ** 2).reshape(target.shape[0], -1), 1).mean()

                if calc_dec_loss:
                    return torch.mean(
                        (w * (predict - target) ** 2).reshape(target.shape[0], -1) * aux_w.detach(), 1
                    ).mean() + self.additional_loss_weight * hf_loss + similarity * self.w_similarity
                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                if multimodal:
                    if calc_dec_loss:
                        camera,range_image = torch.chunk(predict,2,0)
                        kl_distance = self.calc_kl_distance(camera,range_image)
                        similarity = 0.1 * torch.abs(kl_distance - gt_kl_distance)
                        camera_loss,range_image_loss = torch.mean((w*(predict - target)**2).reshape(target.shape[0],-1),1).chunk(2,0)
                        camera_loss = torch.mean(camera_loss)
                        range_image_loss = torch.mean(range_image_loss)
                        loss = camera_loss / camera_loss.detach() + range_image_loss / range_image_loss.detach() + similarity / similarity.detach()
                        return loss,camera_loss,range_image_loss,similarity
                    camera_loss,range_image_loss = torch.mean((w*(predict - target)**2).reshape(target.shape[0],-1),1).chunk(2,0)
                    camera_loss = torch.mean(camera_loss)
                    range_image_loss = torch.mean(range_image_loss)
                    loss = camera_loss / camera_loss.detach() + range_image_loss / range_image_loss.detach()
                else:
                    camera_loss = torch.mean((w*(predict - target)**2).reshape(target.shape[0],-1))
                    loss = camera_loss
                    range_image_loss = camera_loss
                return loss,camera_loss,range_image_loss
        elif self.loss_type == "l1":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.num_frames)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.num_frames)
                bs = target.shape[0] // self.num_frames
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])).abs()
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=1)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.num_frames - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf).abs()).reshape(target.shape[0], -1), 1).mean()
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1), 1
                )
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
            

        


