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
from ldm.models.condition import StreamingSDCondition
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
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"]
        self.loss_type = loss_type
        self.use_additional_loss = use_additional_loss

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.offset_noise_level = offset_noise_level
        self.additional_loss_weight = additional_loss_weight
        
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
            conditioner: StreamingSDCondition,
            input: torch.Tensor,
            batch: Dict
    ):
        cond = conditioner(batch)
        return self._forward(network,denoiser,cond,input)
    

    def _forward(
            self,
            network:nn.Module,
            denoiser:Denoiser,
            cond:Dict,
            x:torch.Tensor,
    ):
        sigmas = self.sigma_sampler(x.shape[0]).to(x)
        
        noise = torch.randn_like(x)
        if self.offset_noise_level > 0.0:
            offset_shape = (x.shape[0],x.shape[1])
            rand_init = torch.randn(offset_shape,device=x.device)
            noise = noise + self.offset_noise_level * append_dims(rand_init,x.ndim)
        sigmas_bc = append_dims(sigmas,x.ndim)
        
        noised_x = self.get_noised_input(sigmas_bc,noise,x)
       
        model_output = denoiser(network,noised_x,sigmas,cond)
        
        w = append_dims(self.loss_weighting(sigmas),x.ndim)
        predict = model_output
        
        input = x
        return self.get_loss(predict,input,w)

    def get_loss(self,predict,target,w,):

        if self.loss_type == "l2":
            if self.use_additional_loss:
                predict_hf = fourier_filter(predict,scale=0.)
                target_hf = fourier_filter(target,scale=0.)
                hf_loss = torch.mean((w*(predict_hf - target_hf) ** 2).reshape(target.shape[0],-1),1).mean()
                return torch.mean((w*(predict-target) ** 2).reshape(target.shape[0],-1),1).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w*(target - predict) ** 2).reshape(target.shape[0],-1),1
                ).mean()
        elif self.loss_type == "l1":
            if self.use_additional_loss:
                predict_hf = fourier_filter(predict,scale=0.)
                target_hf = fourier_filter(target,scale=0.)
                hf_loss = torch.mean((w*(predict_hf - target_hf).abs()).reshape(target.shape[0],-1),1).mean()
                return torch.mean((w*(predict-target).abs()).reshape(target.shape[0],-1),1).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1), 1
                ).mean()
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
            

        


