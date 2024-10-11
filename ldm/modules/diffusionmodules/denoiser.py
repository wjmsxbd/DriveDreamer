from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization
from einops import rearrange
from ldm.modules.diffusionmodules.openaimodel import VideoUNet,UNetModel


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict,movie_len:int = 5,use_actionformer=False):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)
        self.movie_len = movie_len
        self.use_actionformer = use_actionformer

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        x:torch.Tensor,
        range_image:torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        movie_len:int,
        action_former:nn.Module,
        cond_mask:torch.Tensor
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, x.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        if not range_image is None:
            c_skip = torch.cat([c_skip,c_skip],dim=0)
            c_out = torch.cat([c_out,c_out],dim=0)
            c_in = torch.cat([c_in,c_in],dim=0)
            c_noise = torch.cat([c_noise,c_noise],dim=0)
        condition_keys = cond.keys()
        b = x.shape[0] // movie_len
        if 'bev_images' in condition_keys:
            if 'text_emb' in condition_keys:
                text_emb = cond['text_emb']
                text_emb = torch.cat([text_emb,text_emb],dim=0)
            else:
                text_emb = None
            bev_images = cond['bev_images']
            z = torch.cat([x,bev_images],dim=1)
            lidar_z = torch.cat([range_image,bev_images],dim=1)
            actions = rearrange(cond['actions'],'b n c -> (b n) c')
            actions = torch.cat([actions,actions],dim=0)
            boxes_emb = None
        else:
            if not range_image is None:
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = boxes_emb.reshape((b,movie_len)+boxes_emb.shape[1:])
                    boxes_emb = boxes_emb[:,0]
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                if 'hdmap' in condition_keys and 'dense_range_image' in condition_keys:
                    if self.use_actionformer:
                        dense_range_image = cond['dense_range_image']
                        actions = cond['actions']
                        hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                        dense_range_image = dense_range_image.reshape((b,movie_len)+dense_range_image.shape[1:])[:,0]
                        h = action_former(hdmap,boxes_emb,actions,dense_range_image)
                        latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                        latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                        latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
                    else:
                        dense_range_image = cond['dense_range_image']
                        # hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                        actions = cond['actions']
                        # dense_range_image = dense_range_image.reshape((b,movie_len)+dense_range_image.shape[1:])[:,0]
                        # h = action_former(hdmap,boxes_emb,actions,dense_range_image)
                        # latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                        # latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                        # latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
                        latent_hdmap = cond['hdmap']
                        latent_boxes = cond['boxes_emb']
                        latent_dense_range_image = cond['dense_range_image']

                    boxes_emb = torch.cat([latent_boxes,latent_boxes],dim=0)
                    z = torch.cat([x,latent_hdmap],dim=1)
                    lidar_z = torch.cat([range_image,latent_dense_range_image],dim=1)
                    actions = None
                elif not 'hdmap' in condition_keys and not 'dense_range_image' in condition_keys:
                    z = x
                    lidar_z = range_image
                    actions = rearrange(cond['actions'],'b n c -> (b n) c')
                    actions = torch.cat([actions,actions],dim=0)
                else:
                    raise NotImplementedError
            else:
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = boxes_emb.reshape((b,movie_len)+boxes_emb.shape[1:])
                    boxes_emb = boxes_emb[:,0]
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                else:
                    text_emb = None
                actions = cond['actions']
                if 'hdmap' in condition_keys:
                    if self.use_actionformer:
                        hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                        h = action_former(hdmap,boxes_emb,actions)
                        latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                        latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                    else:
                        # hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                        # h = action_former(hdmap,boxes_emb,actions)
                        # latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                        # latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                        latent_hdmap = cond['hdmap']
                        latent_boxes = cond['boxes_emb']
                    z = torch.cat([x,latent_hdmap],dim=1)
                    boxes_emb = latent_boxes
                    lidar_z = None
                else:
                    assert None

        if lidar_z is None:
            input = z
            real_input = x
            cond_mask = cond_mask
            class_label = None
        else:
            input = torch.cat([z,lidar_z])
            classes = torch.tensor([[1.,0.],[0.,1.]],device=x.device,dtype=x.dtype)
            classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)
            class_label = torch.zeros((input.shape[0],4),device=x.device)
            for i in range(input.shape[0]):
                if i < input.shape[0] // 2:
                    class_label[i] = classes_emb[0]
                else:
                    class_label[i] = classes_emb[1]
            cond_mask = torch.cat([cond_mask,cond_mask],dim=0)
            real_input = torch.cat([x,range_image],dim=0)
        if isinstance(network,UNetModel):
            return (
                network(input*c_in,c_noise,y=class_label,boxes_emb=boxes_emb,text_emb=text_emb,cond_mask=cond_mask,actions=actions) * c_out
                + real_input * c_skip
            )
        else:
            return (
                network(input*c_in,c_noise,y=class_label,context=boxes_emb,cond_mask=cond_mask,num_frames=self.movie_len) * c_out + real_input * c_skip
            )


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise