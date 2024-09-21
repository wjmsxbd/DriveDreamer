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
            actionformer:nn.Module
    ):
        return self._forward(network,denoiser,cond,x,range_image,actionformer)
    
    def _forward(
            self,
            network:nn.Module,
            denoiser:Denoiser,
            cond:Dict,
            x:torch.Tensor,
            range_image:torch.Tensor,
            actionformer:nn.Module,
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
        noised_range_image = self.get_noised_input(sigmas_bc,noise,range_image)
        
        model_output = denoiser(network,noised_x,noised_range_image,sigmas,cond,self.movie_len,actionformer,cond_mask)
        x_rec,range_image_rec = torch.chunk(model_output,2,0)
        sigmas = torch.cat([sigmas,sigmas],dim=0)
        w = append_dims(self.loss_weighting(sigmas),x.ndim)
        if self.replace_cond_frames:
            predict_x = x_rec * append_dims(1 - cond_mask,x.ndim) + x * append_dims(cond_mask,x.ndim)
            predict_range = range_image_rec * append_dims(1 - cond_mask,x.ndim) + range_image_rec * append_dims(cond_mask,x.ndim)
            predict = torch.cat([predict_x,predict_range],dim=0)
        else:
            predict = model_output
        input = torch.cat([x,range_image],dim=0)
        return self.get_loss(predict,input,w,cond)
    
    def get_loss(self,predict,target,w,cond):
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

                # if self.depth_estimator is not None:
                #     predict_x,predict_lidar = torch.chunk(predict,2,0)
                #     predict_lidar = (torch.clip(predict_lidar,-1,1) + 1) / 2.
                #     mask = predict_lidar > 0.05
                #     predict_lidar = (predict_lidar * mask) * 255.
                #     indices = [torch.nonzero(predict_lidar[i]) for i in range(predict_lidar.shape[0])]
                #     for batch_id in range(predict_lidar.shape[0]):
                #         indice = indices[batch_id]
                #         points = torch.zeros((len(indice),3))
                #         for idx in range(len(indice)):
                #             vi,ui = indice[idx]
                #             phi = ((1+ui)/self.img_size[1] - 1) * torch.pi / 6
                #             theta = self.theta_up - (self.theta_up - self.theta_down) * ((vi+1) - 1./2) / self.img_size[0]
                #             r = predict_lidar[batch_id,vi,ui,0]
                #             point_x = r * torch.cos(theta) * torch.sin(phi)
                #             point_y = r * torch.cos(theta) * torch.cos(phi)
                #             point_z = r * torch.sin(theta)
                #             points[idx] = torch.tensor([point_x,point_y,point_z]).to(torch.float32)
                #         points = rearrange(points,'n c -> c n')
                #         pc = LidarPointCloud(points=points)
                #         pc.rotate(Quaternion(cond['Lidar_TOP_record']['rotation']).rotation_matrix)
                #         pc.translate(np.array(cond['Lidar_TOP_record']['translation']))

                #         pc.rotate(Quaternion(cond['Lidar_TOP_poserecord']['rotation']).rotation_matrix)
                #         pc.translate(np.array(cond['Lidar_TOP_poserecord']['translation']))

                #         pc.translate(-np.array(cond['cam_poserecord']['translation']))
                #         pc.rotate(Quaternion(cond['cam_poserecord']['rotation']).rotation_matrix.T)

                #         pc.translate(-np.array(cond['cam_front_record']['translation']))
                #         pc.rotate(Quaternion(cond['cam_front_record']['rotation']).rotation_matrix.T)
                #         points = pc.points
                #         depths = points[2,:]
                #         points = view_points(pc.points[:3,:],np.array(cond['cam_front_record']['camera_intrinsic']),normalize=True)
                #         mask = np.ones(depths.shape[1],dtype=bool)
                #         mask = np.logical_and(mask, depths > 1)
                #         mask = np.logical_and(mask,points[0,:] > 0)
                #         mask = np.logical_and(mask,points[0,:] < 1600)
                #         mask = np.logical_and(mask,points[1,:] > 0)
                #         mask = np.logical_and(mask,points[1,:] < 1200)
                #         points = points[:,mask]
                #         depths = depths[mask]
                #         points[0,:] = points[0,:] / 1600 * self.img_size[1]
                #         points[1,:] = points[1,:] / 1200 * self.img_size[0]
                #         points = points.to(torch.uint8)

                #         pred_depth = self.depth_estimator(predict_x)
                #         sampled_depth = pred_depth[points]
                #         loss1 = (sampled_depth - depths) ** 2

                

                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1), 1
                )
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
            




