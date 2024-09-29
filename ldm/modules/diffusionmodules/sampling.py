"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ldm.modules.diffusionmodules.sampling_utils import to_d
from ldm.util import append_dims, default, instantiate_from_config


class BaseDiffusionSampler:
    def __init__(
            self,
            discretization_config: Union[Dict, ListConfig, OmegaConf],
            num_steps: Union[int, None] = None,
            guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
            verbose: bool = False,
            device: str = "cuda"
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(guider_config)
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, cond_x,cond_range_image, denoiser, sigma, cond, cond_mask, uc):
        denoised = denoiser(*self.guider.prepare_inputs(cond_x,cond_range_image, sigma, cond, cond_mask, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling Setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps"
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EulerEDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, cond_x,cond_range_image, cond, cond_mask=None, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(cond_x) * self.s_noise
            if cond_range_image is None:
                cond_x = cond_x + eps * append_dims(sigma_hat ** 2 - sigma ** 2,cond_x.ndim) ** 0.5
            else:
                cond_x = cond_x + eps * append_dims(sigma_hat ** 2 - sigma ** 2,cond_x.ndim) ** 0.5
                cond_range_image = cond_range_image + eps * append_dims(sigma_hat ** 2 - sigma ** 2,cond_range_image.ndim) ** 0.5

        denoised = self.denoise(cond_x,cond_range_image, denoiser, sigma_hat, cond, cond_mask, uc)
        if not cond_range_image is None:
            x = torch.cat([cond_x,cond_range_image],dim=0)
            sigma_hat = torch.cat([sigma_hat,sigma_hat],dim=0)
            next_sigma = torch.cat([next_sigma,next_sigma],dim=0)
        else:
            x = cond_x
        
        # sigma_hat = torch.cat([sigma_hat,sigma_hat],dim=0)
        # next_sigma = torch.cat([next_sigma,next_sigma],dim=0)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
            self,
            denoiser,
            x,  # x is randn
            cond,
            uc=None,
            cond_x=None,
            cond_range_image=None,
            cond_mask=None,
            num_steps=None
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
        replace_cond_frames = cond_mask is not None and cond_mask.any()
        x_x,range_x = x,x
        temp = cond_x
        eps = 1e-7
        print(self.get_sigma_gen(num_sigmas))
        for i in self.get_sigma_gen(num_sigmas):
            if replace_cond_frames:
                cond_x = x_x * append_dims(1 - cond_mask, x.ndim) + cond_x * append_dims(cond_mask, cond_x.ndim)
                if not cond_range_image is None:
                    cond_range_image = range_x * append_dims(1 - cond_mask,x.ndim) + cond_range_image * append_dims(cond_mask,cond_range_image.ndim)
                else:
                    cond_range_image = None
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                cond_x,
                cond_range_image,
                cond,
                cond_mask,
                uc,
                gamma
            )
            if not cond_range_image is None:
                x_x,range_x = torch.chunk(x,2,0)
        if replace_cond_frames:
            # assert cond_range_image is None
            if cond_range_image is None:
                x_rec = x
                x_rec = x_rec * append_dims(1-cond_mask,x_rec.ndim) + cond_x * append_dims(cond_mask,cond_x.ndim)
                x = x_rec
            else:
                x_rec,range_rec = torch.chunk(x,2,dim=0)
                x_rec = x_rec * append_dims(1-cond_mask,x_rec.ndim) + cond_x * append_dims(cond_mask,cond_x.ndim)
                range_rec = range_rec * append_dims(1-cond_mask,range_rec.ndim) + cond_range_image * append_dims(cond_mask,range_rec.ndim)
                x = torch.cat([x_rec,range_rec],dim=0)
        print(torch.max(x[0] - temp[0]))
        assert torch.max(x[0] - temp[0]) < eps
        return x