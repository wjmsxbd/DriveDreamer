import os
import torch
import torch.nn as nn
import numpy as np
from einops import repeat

def make_beta_schedule(schedule,n_timestep,linear_start=1e-4,linear_end=2e-2,cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5,linear_end ** 0.5,n_timestep,dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep+1,dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1+cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas,a_min=0,a_max=0.999)
    elif schedule == 'sqrt_linear':
        betas = torch.linspace(linear_start,linear_end,n_timestep,dtype=torch.float64)
    elif schedule == 'sqrt':
        betas = torch.linspace(linear_start,linear_end,n_timestep,dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unkown.")
    return betas.numpy()

def extract_into_tensor(a,t,x_shape):
    b,*_ = t.shape
    out = a.gather(-1,t)
    return out.reshape(b,*((1,) * (len(x_shape) - 1)))

#return repeat noise or noise
def noise_like(shape,device,repeat=False):
    repeat_noise = lambda: torch.randn((1,*shape[1:]),device=device).repeat(shape[0],*((1,) * (len(shape)-1)))
    noise = lambda:torch.randn(shape,device=device)
    return repeat_noise() if repeat else noise()