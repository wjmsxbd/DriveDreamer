"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
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
from typing import Iterable,List,Union,Optional
from ldm.modules.diffusionmodules.openaimodel import UNetModel,VideoUNet

__conditioning_keys__ = {'concat':'c_concat',
                         'crossattn':'c_crossattn',
                         'adm':'y'}

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1,r2,shape,device):
    return (r1-r2) * torch.rand(*shape,device=device) + r2


class DiffusionWrapper(pl.LightningModule):
    def __init__(self,model_config):
        super().__init__()
        self.model = instantiate_from_config(model_config)

    def get_last_layer(self):
        return self.model.get_last_layer()

    def forward(self,
            x:torch.Tensor,
            timesteps:torch.Tensor,
            context: Optional[torch.Tensor]=None,
            y:Optional[torch.Tensor]=None,
            time_context:Optional[torch.Tensor]=None,
            cond_mask:Optional[torch.Tensor] = None,
            num_frames: Optional[int] = None,
            x_0 = None,
            range_image_0=None,):
        x_0 = x_0.unsqueeze(1).repeat(1,num_frames,1,1,1)
        range_image_0 = range_image_0.unsqueeze(1).repeat(1,num_frames,1,1,1)
        x_0 = torch.cat([x_0,range_image_0],dim=0)
        x_0 = rearrange(x_0,'b n c h w -> (b n) c h w').contiguous()
        x[:,4:] += x_0
        return self.model(x,timesteps,context,y,time_context,cond_mask,num_frames)
        

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=False,
                 first_stage_key="reference_image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,# weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",# all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps","x0"],'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}:Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = instantiate_from_config(unet_config)#.eval()
        count_params(self.model,verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(self.model_ema.m_name2s_name)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path,ignore_keys=ignore_keys,only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas,beta_schedule=beta_schedule,timesteps=timesteps,
                               linear_start=linear_start,linear_end=linear_end,cosine_s=cosine_s)

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init,size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar,requires_grad=True)


    def register_schedule(self,given_betas=None,beta_schedule="linear",timesteps=1000,
                          linear_start=1e-4,linear_end=2e-2,cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule,timesteps,linear_start,linear_end,cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas,axis=0)
        alphas_cumprod_prev = np.append(1.,alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps,'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor,dtype=torch.float32)

        self.register_buffer('betas',to_torch(betas))
        self.register_buffer('alphas_cumprod',to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',to_torch(np.log(np.maximum(posterior_variance,1e-20))))
        self.register_buffer('posterior_mean_coef1',to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        #TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights',lvlb_weights,persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self,context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            #TODO: when call
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def init_from_ckpt(self,path,ignore_keys=list(),only_model=False,load_from_ema=False):
        sd = torch.load(path,map_location='cpu')
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        if load_from_ema:
            s_name2m_name = dict(zip(self.model_ema.m_name2s_name.values(),self.model_ema.m_name2s_name.keys()))
            for k in list(sd.keys()):
                if k.startswith('model_ema'):
                    # print(k)
                    v = sd[k]
                    k = k[len('model_ema.'):]
                    if k in s_name2m_name.keys():
                        k = 'model.' + s_name2m_name[k]
                        # print(f"after:{k}")
                        sd[k] = v
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

    def q_mean_variance(self,x_start,t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self,x_start,x_t,t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1,t,x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2,t,x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance,t,x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped,t,x_t.shape)
        return posterior_mean,posterior_variance,posterior_log_variance_clipped

    def p_mean_variance(self,x,t,clip_denoised:bool):
        model_out = self.model(x,t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x,t=t,noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.,1.)

        model_mean,posterior_variance,posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean,posterior_variance,posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self,x,t,clip_denoised=True,repeat_noise=False):
        b,*_,device = *x.shape,x.device
        model_mean,_,model_log_variance = self.p_mean_variance(x=x,t=t,clip_denoised=clip_denoised)
        noise = noise_like(x.shape,device,repeat_noise)
        # no noise when t==0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp()*noise

    @torch.no_grad()
    def p_sample_loop(self,shape,return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape,device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0,self.num_timesteps)),desc='Sampling t',total=self.num_timesteps):
            img = self.p_sample(img,torch.full((b,),i,device=device,dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img,intermediates
        return img
    
    @torch.no_grad()
    def sample(self,batch_size=16,return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size,channels,*image_size),return_intermediates=return_intermediates)
    
    def q_sample(self,x_start,t,noise=None):
        noise = default(noise,lambda:torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod,t,x_start.shape) * x_start + 
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape) * noise)
    
    def get_loss(self,pred,target,mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target,pred)
            else:
                loss = torch.nn.functional.mse_loss(target,pred,reduction='none')
        else:
            raise NotImplementedError("unkown loss type '{loss_type}'")
        return loss
    
    def p_losses(self,x_start,t,noise=None):
        noise = default(noise,lambda:torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
        model_out = self.model(x_noisy,t)
        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        loss = self.get_loss(model_out,target,mean=False).mean(dim=[1,2,3])

        log_prefix = "train" if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb':loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        
        loss_dict.update({f'{log_prefix}/loss':loss})
        return loss,loss_dict
    
    def forward(self,x,*args,**kwargs):
        t = torch.randint(0,self.num_timesteps,(x.shape[0],),device=self.device).long()
        return self.p_losses(x,t,*args,**kwargs)
    
    def get_input(self,batch,k):
        x = batch[k]
        # if len(x.shape) == 4:
        #     x = x.unsqueeze(1)
        x = rearrange(x,'b n h w c -> (b n) c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def shared_step(self,batch):
        x = self.get_input(batch,self.first_stage_key)
        loss,loss_dict = self(x)
        return loss,loss_dict
    
    def training_step(self,batch,batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        _,loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _,loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key+'_ema':loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self,*args,**kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self,samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_row)
        return denoise_grid
    
    @torch.no_grad()
    def log_images(self,batch,N=8,n_row=2,sample=True,return_keys=None,**kwargs):
        log = dict()
        x = self.get_input(batch,self.first_stage_key)
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        x = x.to(self.device)[:N]
        log['inputs'] = x

        #get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]),'1 -> b',b = n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
                diffusion_row.append(x_noisy)

        log['diffusion_row'] = self._get_rows_from_list(diffusion_row)

        if sample:
            #get denoise row
            with self.ema_scope("Plotting"):
                samples,denoise_row = self.sample(batch_size=N,return_intermediates=True)
            
            log['samples'] = samples
            log['denoise_row'] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                return log
            else:
                return {key:log[key] for key in return_keys}
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params,lr=lr)
        return opt

#TODO:whether need split image
class AutoDM(DDPM):
    def __init__(self,
                 base_learning_rate,
                 ref_img_config,
                 hdmap_config,
                 box_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unet_trainable=True,
                 *args,**kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond,1)
        self.scale_by_std = scale_by_std
        self.learning_rate = base_learning_rate
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = 'concat' if concat_mode else 'crossattn'
        # if cond_stage_config == '__is_unconditional__':
        #     conditioning_key = None
        super().__init__(conditioning_key=None,*args,**kwargs)
        unet_config = kwargs.pop('unet_config',[])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.unet_trainable = unet_trainable
        try:
            self.num_downs = len(ref_img_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor',torch.tensor(scale_factor))
        self.ref_img_encoder = instantiate_from_config(ref_img_config)
        self.ref_img_encoder.eval()
        for param in self.ref_img_encoder.parameters():
            param.requires_grad = False
        self.hdmap_encoder = instantiate_from_config(hdmap_config)
        self.hdmap_encoder.eval()
        for param in self.hdmap_encoder.parameters():
            param.requires_grad = False
        self.box_encoder = instantiate_from_config(box_config)
        #self.instantiate_cond_stage(cond_stage_config)
        # self.split_input_params = {
        #     "ks":(1024,1024),
        #     "stride":(512,512),
        #     "vqf": 64,
        #     "patch_distributed_vq": True,
        #     "tie_breaker":False,
        #     "patch_distributed_vq": True,
        #     "tie_braker": False,
        #     "clip_max_weight": 0.5,
        #     "clip_min_weight": 0.01,
        #     "clip_max_tie_weight": 0.5,
        #     "clip_min_tie_weight": 0.01
        # }
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False
        if unet_config['params']['ckpt_path'] is not None:
            # self.model.from_pretrained_model(unet_config['params']['ckpt_path'])
            self.from_pretrained_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'])
            self.restarted_from_ckpt = True

    def from_pretrained_model(self,model_path,ignore_keys):
        print("now from pretrained model")
        if not os.path.exists(model_path):
            raise RuntimeError(f'{model_path} does not exist')
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        my_model_state_dict = self.state_dict()
        keys = list(sd.keys())
        # for param in state_dict.keys():
        #     if param not in my_model_state_dict.keys():
        #         print("Missing Key:"+str(param))
        #     else:
        #         for ignore_key in ignore_keys:
        #             if not param.startswith(ignore_key):
        #                 params[param] = state_dict[param]
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
        #torch.save(my_model_state_dict,'./my_model_state_dict.pt')
        self.load_state_dict(sd,strict=False)

    def make_cond_schedule(self,):
        self.cond_ids = torch.full(size=(self.num_timesteps,),fill_value=self.num_timesteps - 1,dtype=torch.long)
        ids = torch.round(torch.linspace(0,self.num_timesteps - 1 ,self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids
    
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self,batch,batch_idx,current_epoch):
        #only for very first batch
        if self.scale_by_std and current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            #set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch,self.first_stage_key)
            encoder_posterior = self.encode_first_stage(x,self.ref_img_encoder)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor',1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
            return
        

    def instantiate_cond_stage(self,config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.clip = model.eval()
            self.clip.train = disabled_train
            for param in self.clip.parameters():
                param.requires_grad = False
            
        else:
            model = instantiate_from_config(config)
            self.clip = model

    def _get_denoise_row_from_list(self,samples,desc='',force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples,desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row) #n_log_step n_row C H W
        denoise_grid = rearrange(denoise_row,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_row)
        return denoise_grid
        
    def get_first_stage_encoding(self,encoder_posterior):
        if isinstance(encoder_posterior,DiagonalGuassianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior,torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def get_learned_conditioning(self,c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model,'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c,DiagonalGuassianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model,self.cond_stage_forward)
            c = getattr(self.cond_stage_model,self.cond_stage_forward)(c)
        return c

    def meshgrid(self,h,w):
        y = torch.arange(0,h).view(h,1,1).repeat(1,w,1)
        x = torch.arange(0,w).view(1,w,1).repeat(h,1,1)
        arr = torch.cat([y,x],dim=-1)
        return arr
    
    def delta_border(self,h,w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h-1,w-1]).view(1,1,2)
        arr = self.meshgrid(h,w) / lower_right_corner
        dist_left_up = torch.min(arr,dim=-1,keepdims=True)[0]
        dist_right_down = torch.min(1-arr,dim=-1,keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up,dist_right_down],dim=-1),dim=-1)[0]
        return edge_dist
    
    def get_weighting(self,h,w,Ly,Lx,device):
        weighting = self.delta_border(h,w)
        weighting = torch.clip(weighting,self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"],)
        weighting = weighting.view(1,h*w,1).repeat(1,1,Ly*Lx).to(device)

        if self.split_input_params["tie_breaker"]:
            L_weighting = self.delta_border(Ly,Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            L_weighting = L_weighting.view(1,1,Ly*Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def training_step(self,batch,batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
            fold_params2 = dict(kernel_size=(kernel_size[0]*uf,kernel_size[0]*uf),
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
            fold_params2 = dict(kernel_size=(kernel_size[0] // df,kernel_size[0] // df),
                                dilation=1,padding=0,
                                stride=(stride[0] // df,stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df,x.shape[3] // df),**fold_params2)
            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1,1,h//df,w//df)
            weighting = weighting.view((1,1,kernel_size[0]//df,kernel_size[1]//df,Ly*Lx))
        else:
            raise NotImplementedError
        return fold,unfold,normalization,weighting
    
    def get_first_stage_encoding(self,encoder_posterior):
        if isinstance(encoder_posterior,DiagonalGuassianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior,torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f'encoder_posterior of type {type(encoder_posterior)} not yet implemented')
        return self.scale_factor * z

    #check whether need / 255.
    def encode_first_stage(self,x,encoder):
        if hasattr(self,'split_input_params'):
            if self.split_input_params['patch_distributed_vq']:
                ks = self.split_input_params["ks"] #eg. (128,128)
                stride = self.split_input_params["stride"] # eg. (64,64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
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
                return encoder.encode(x)
        else:
            return encoder.encode(x)
        

    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False):
        ref_img = super().get_input(batch,'reference_image')
        hdmap = super().get_input(batch,'HDmap')
        boxes = batch['3Dbox']
        boxes_category = batch['category']
        text = batch['text']
        boxes = rearrange(boxes,'b n x c -> (b n) x c')
        boxes_category = rearrange(boxes_category,'b n x c -> (b n) x c')
        text = rearrange(text,'b n c -> (b n) c').unsqueeze(1)
        x = ref_img
        z = self.encode_first_stage(x,self.ref_img_encoder)
        # z = self.ref_img_encoder.encode(x)
        z = self.get_first_stage_encoding(z)
        out = [z]
        c = {}
        c['hdmap'] = hdmap.to(torch.float32)
        c['boxes'] = boxes.to(torch.float32)
        c['category'] = boxes_category.to(torch.float32)
        c['text'] = text.to(torch.float32)
        if return_first_stage_outputs:
            x_rec = self.decode_first_stage(z)
            out.extend([x,x_rec])
        out.append(c)
        return out
        # ref_img = super().get_input(batch,'reference_image')
        # hdmap = super().get_input(batch,'HDmap')
        # boxes = batch['3Dbox'].to(torch.float32).to(self.device)
        # boxes_category = batch['category']
        # text = batch['text']
        # x_embed = self.encode_first_stage(ref_img,self.ref_img_encoder)
        # hdmap_emb = self.encode_first_stage(hdmap,self.hdmap_encoder)
        # x_embed = self.get_first_stage_encoding(x_embed).detach()
        # hdmap_emb = self.get_first_stage_encoding(hdmap_emb).detach()
        
        # if self.use_positional_encodings:
        #     boxes_emb = rearrange(boxes_emb,'b n h w -> b (n h) w')
        #     # TODO:whether need add postional encoder
        #     # boxes_postional_encoder = PositionalEncoder(d_model=boxes_emb.shape[2],seq_len=boxes_emb.shape[1])
        #     # boxes_emb = boxes_postional_encoder(boxes_emb)
        #     # boxes_category_postional_encoder = PositionalEncoder(d_model=boxes_category.shape[2],seq_len=boxes_category.shape[1])
        #     # boxes_category = boxes_category_postional_encoder(boxes_category)
        #     boxes_emb = self.box_encoder.encode(boxes,boxes_category)
        # else:
        #     boxes_emb = boxes
        # c = {'hdmap_emb':hdmap_emb,
        #     'boxes_emb':boxes_emb,
        #      'box_clip':boxes_category,
        #      'text':text}
        # out = [x_embed,c]
        # if return_first_stage_outputs:
        #     xrec = self.decode_first_stage(x_embed)
        #     out.extend([ref_img,xrec])
        # return out
        
    # c = {'hdmap':...,"boxes":...,'category':...,"text":...}
    def shared_step(self,batch,**kwargs):
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        hdmap = super().get_input(batch,'HDmap') # (b n c h w)
        x = ref_img.to(torch.float32)
        x = self.get_first_stage_encoding(self.encode_first_stage(x,self.ref_img_encoder))
        c = {}
        c['hdmap'] = hdmap.to(torch.float32)
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d')
        boxes_category = rearrange(batch['category'],'b n c d -> (b n) c d')
        c['boxes'] = boxes.to(torch.float32)
        c['category'] = boxes_category.to(torch.float32)
        c['text'] = rearrange(batch['text'],'b n d -> (b n) d').unsqueeze(1).to(torch.float32)
        loss,loss_dict = self(x,c)
        return loss,loss_dict

    def forward(self,x,c,*args,**kwargs):
        t = torch.randint(0,self.num_timesteps,(x.shape[0],),device=self.device).long()#
        return self.p_losses(x,c,t,*args,**kwargs)

    def apply_model(self,x_noisy,t,cond,return_ids=False):
        #TODO:wrong
        if hasattr(self,"split_input_params"):
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold,unfold,normalization,weighting = self.get_fold_unfold(x_noisy,ks,stride)

            z = unfold(x_noisy) # (bn, nc*prod(**ks),L)
            # Reshape to img shape
            z = z.view((z.shape[0],-1,ks[0],ks[1],z.shape[-1])) # (bn,nc,ks[0],ks[1],L)
            z_list = [z[:,:,:,:,i] for i in range(z.shape[-1])]

            #TODO:process condition
            #HDmap
            fold_hdmap,unfold_hdmap,normalization_hdmap,weighting_hdmap = self.get_fold_unfold(cond['hdmap'],ks,stride)
            hdmap_split = unfold_hdmap(cond['hdmap'])
            hdmap_split = hdmap_split.view((hdmap_split.shape[0],-1,ks[0],ks[1],hdmap_split.shape[-1]))
            hdmap_list = [hdmap_split[:,:,:,:,i] for i in range(hdmap_split.shape[-1])]
            #boxes
            boxes_list = []
            boxes_category_list = []
            col_h = np.arange(0,h-ks[0],stride[0])
            row_w = np.arange(0,w-ks[1],stride[1])
            boxes = cond['boxes'].reshape(x_noisy.shape[0],-1,2,8)
            boxes_category = cond['category']
            for i in range(len(col_h)):
                for j in range(len(row_w)):
                    min_x = row_w[j]
                    min_y = col_h[i]
                    max_x = row_w[j] + ks[1]
                    max_y = col_h[i] + ks[0]
                    visible = np.logical_and(boxes[:,:,0,:]>=min_x,boxes[:,:,0,:]<=max_x)
                    visible = np.logical_and(visible,boxes[:,:,1,:]>=min_y)
                    visible = np.logical_and(visible,boxes[:,:,1,:]<=max_y)
                    visible = torch.any(visible,dim=-1,keepdim=True).bool()
                    boxes_mask = visible.unsqueeze(3).repeat(1,1,2,8)
                    patch_boxes = torch.masked_fill(boxes,boxes_mask,0).reshape(x_noisy.shape[0],-1,16)
                    boxes_list.append(patch_boxes)
                    category_mask = visible.repeat(1,1,768)
                    patch_category = torch.masked_fill(boxes_category,category_mask,0)
                    boxes_category_list.append(patch_category)
            output_list = []
            for i in range(z.shape[-1]):
                z = z_list[i]
                z = self.get_first_stage_encoding(self.encode_first_stage(z,self.ref_img_encoder))
                hdmap = hdmap_list[i]
                hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
                z = torch.cat([z,hdmap],dim=1)
                boxes = boxes_list[i]
                
                #boxes = rearrange(boxes,'b n c -> (b n) c')
                boxes_category = boxes_category_list[i]
                boxes_emb = self.box_encoder(boxes,boxes_category)
                text_emb = cond['text']
                output = self.model(z,t,boxes_emb,text_emb)
                output_list.append(output)
            o = torch.stack(output_list,axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0],-1,o.shape[-1])) #(bn,nc*ks[0]*ks[1],L)
            # stitch crops together
            x_recon = fold(0) / normalization
        else:
            hdmap = self.get_first_stage_encoding(self.encode_first_stage(cond['hdmap'],self.hdmap_encoder))
            boxes_emb = self.box_encoder(cond['boxes'],cond['category'])
            text_emb = cond['text']
            z = torch.cat([x_noisy,hdmap],dim=1)
            #TODO:need reshape z
            x_recon = self.model(z,t,boxes_emb=boxes_emb,text_emb=text_emb)
        return x_recon


    
    def p_losses(self,x_start,cond,t,noise=None):
        noise = default(noise,lambda:torch.randn_like(x_start)).to(x_start.device)        
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
        model_output = self.apply_model(x_noisy,t,cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'eps':
            target = noise
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output,target,mean=False).mean([1,2,3])
        loss_dict.update({f'{prefix}/loss_simple':loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma':loss.mean()})
            loss_dict.update({'logvar':self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output,target,mean=False).mean(dim=[1,2,3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb':loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss':loss})

        return loss,loss_dict
    
    def decode_first_stage(self,z,predict_cids=False,force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.ref_img_encoder.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1. / self.scale_factor * z

        if hasattr(self,"split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
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
            
                output_list = [self.ref_img_encoder.decode(z[:, :, :, :, i])
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
                return self.ref_img_encoder.decode(z)
        else:
            if isinstance(self.ref_img_encoder,VQModelInterface):
                return self.ref_img_encoder.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.ref_img_encoder.decode(z)
    
    def differentiable_decode_first_stage(self,z,predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.ref_img_encoder.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1. / self.scale_factor * z

        if hasattr(self,"split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
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
            
                output_list = [self.ref_img_encoder.decode(z[:, :, :, :, i])
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
                return self.ref_img_encoder.decode(z)
        else:
            if isinstance(self.ref_img_encoder,VQModelInterface):
                return self.ref_img_encoder.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.ref_img_encoder.decode(z)
    
    def _rescale_annotations(self,bboxes,crop_coordinates):
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2],0,1)
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3],0,1)
            w = min(bbox[2] / crop_coordinates[2] , 1 - x0)
            h = min(bbox[3] / crop_coordinates[3] , 1 - y0)
            return x0,y0,w,h
        return [rescale_bbox(b) for b in bboxes]
    
    def _predict_eps_from_xstart(self,x_t,t,pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod,t,x_t.shape) * x_t - pred_xstart) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape) 
    
    def _prior_bpd(self,x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1]*batch_size,device=x_start.device)
        qt_mean,_,qt_log_variance = self.q_mean_variance(x_start,t)
        kl_prior = normal_kl(mean1=qt_mean,logvar1=qt_log_variance,
                             mean2=0.0,logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        if self.unet_trainable:
            print("add unet model parameters into optimizers")
            for name,param in self.model.named_parameters():
                # if 'temporal' in param or 'gated' in param:
                #     print(f"model:add {param} into optimizers")
                #     params.append(param)
                # if 'gated' in name or 'diffusion_model.input_blocks.0.0' in name or 'diffusion_model.out.2' in name:
                #     print(f"model:add {name} into optimizers")
                #     params.append(param)
                # elif 'transoformer'
                # elif not 'temporal' in name:
                #     param.requires_grad = False
                # else:
                #     # assert param.requires_grad == True
                #     param.requires_grad = False
                # if 'gated' in name or 'diffusion_model.input_blocks.0.0' in name or 'diffusion_model.out.2' in name :
                #     print(f"model:add {name} into optimizers")
                #     params.append(param)
                # elif not 'transformer_blocks' in name:
                #     param.requires_grad=False
                if not 'temporal' in name and 'diffusion_model' in name:
                    print(f"model:add {name} into optimizers")
                    params.append(param)
                
        if self.cond_stage_trainable:
            print("add encoder parameters into optimizers")
            params = params  + list(self.hdmap_encoder.parameters()) + list(self.box_encoder.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
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
        
    def p_mean_variance(self,x,c,t,clip_denoised:bool,return_codebook_ids=False,quantize_denoised=False,
                        return_x0=False,score_corrector=None,corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x,t_in,c,return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self,model_out,x,t,c,**corrector_kwargs)

        if return_codebook_ids:
            model_out,logits = model_out
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x,t=t,noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1.,1.)
        # if quantize_denoised:
        #     x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean,posterior_variance,posterior_log_variance = self.q_posterior(x_start=x_recon,x_t=x,t=t)
        if return_codebook_ids:
            return model_mean,posterior_variance,posterior_log_variance,logits
        elif return_x0:
            return model_mean,posterior_variance,posterior_log_variance,x_recon
        else:
            return model_mean,posterior_variance,posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self,x,c,t,clip_denoised=False,repeat_noise=False,
                 return_codebook_ids=False,quantize_denoised=False,return_x0=False,
                 temperature=1.,noise_dropout=0.,score_corrector=None,corrector_kwargs=None):
        b,*_,device = *x.shape,x.device
        outputs = self.p_mean_variance(x=x,c=c,t=t,clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean,_,model_log_variance,x0 = outputs
        else:
            model_mean,_,model_log_variance = outputs
        
        noise = noise_like(x.shape,device,repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise,p=noise_dropout)
        #no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
    @torch.no_grad()
    def progressive_denoising(self,cond,shape,verbose=True,callback=None,quantize_denoised=False,
                              img_callback=None,mask=None,x0=None,temperature=1.,noise_dropout=0.,
                              score_corrector=None,corrector_kwargs=None,batch_size=None,x_T=None,start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
                    range(0,timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        
        for i in iterator:
            ts = torch.full((b,),i,device=self.device,dtype=torch.long)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,t=tc,noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0,ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback:img_callback(img,i)
        return img,intermediates
    
    def p_sample_loop(self,cond,shape,return_intermediates=False,
                      x_T=None,verbose=True,callback=None,timesteps=None,quantize_denoised=False,
                      mask=None,x0=None,img_callback=None,start_T=None,log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=device)
        else:
            img = x_T
        
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        
        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Sampling t',total=timesteps) if verbose else reversed(
            range(0,timesteps))
        
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        
        for i in iterator:
            ts = torch.full((b,),i,device=device,dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if return_intermediates:
            return img,intermediates
        return img
    
    @torch.no_grad()
    def sample(self,cond,batch_size=16,return_intermediates=False,x_T=None,
               verbose=True,timesteps=None,quantize_denoised=False,
               mask=None,x0=None,shape=None,**kwargs):
        if shape is None:
            shape = (batch_size,self.channels) + tuple(self.image_size)
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]
        
        return self.p_sample_loop(cond,shape,return_intermediates=return_intermediates,
                                  x_T=x_T,verbose=verbose,timesteps=timesteps,
                                  quantize_denoised=quantize_denoised,mask=mask,x0=x0)
    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.image_size)
            samples,intermediates = ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,intermediates
    
    @torch.no_grad()
    def log_images(self,batch,N=8,n_row=4,sample=True,ddim_steps=200,ddim_eta=1.,return_keys=None,
                   quantize_denoised=True,inpaint=True,plot_denoise_rows=False,plot_progressive_rows=True,
                   plot_diffusion_rows=True,**kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z,x,x_rec,c = self.get_input(batch,return_first_stage_outputs=True)

        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['inputs'] = x
        log['reconstruction'] = x_rec
        # log['hdmap'] = c['hdmap']

        if plot_diffusion_rows:
            #get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]),'1 -> b',b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start,t=t,noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row) # n_log_step,n_row,C,H,W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log['diffusion_row'] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples,z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log['denoise_row'] = denoise_grid
            
            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
            #         self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
            #                                                  ddim_steps=ddim_steps,eta=ddim_eta,
            #                                                  quantize_denoised=True)
            #         # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
            #         #                                      quantize_denoised=True)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

            if inpaint:
                #make a simple center square
                b,h,w = z.shape[0],z.shape[2],z.shape[3]
                mask = torch.ones(N,h,w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps,x0=z[:N],mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log['samples_inpainting'] = x_samples
                log["mask"] = mask

                #outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps,x0=z[:N],mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples
            if plot_progressive_rows:
                with self.ema_scope("Plotting Progressives"):
                    img, progressives = self.progressive_denoising(c,
                                                                shape=(self.channels, self.image_size[0],self.image_size[1]),
                                                                batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row
            if return_keys:
                if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                    return log
                else:
                    return {key:log[key] for key in return_keys}
            return log

class AutoDM_PretrainedAutoEncoder(DDPM):
    def __init__(self,
                 base_learning_rate,
                 ref_img_config,
                 hdmap_config,
                 box_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unet_trainable=True,
                 split_input_params=None,
                 movie_len = 1,
                 range_image_config = None,
                 downsample_img_size = None,
                 *args,**kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond,1)
        self.scale_by_std = scale_by_std
        self.learning_rate = base_learning_rate
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = 'concat' if concat_mode else 'crossattn'
        # if cond_stage_config == '__is_unconditional__':
        #     conditioning_key = None
        super().__init__(conditioning_key=None,*args,**kwargs)
        unet_config = kwargs.pop('unet_config',[])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.unet_trainable = unet_trainable
        try:
            self.num_downs = len(ref_img_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor',torch.tensor(scale_factor))
        self.first_stage_model = instantiate_from_config(ref_img_config)
        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        self.hdmap_encoder = instantiate_from_config(hdmap_config)
        # self.hdmap_encoder.eval()
        # for param in self.hdmap_encoder.parameters():
        #     param.requires_grad = False
        self.box_encoder = instantiate_from_config(box_config)
        if not range_image_config is None:
            self.range_image_encoder = instantiate_from_config(range_image_config)
        #self.instantiate_cond_stage(cond_stage_config)
        if not split_input_params is None:
            self.split_input_params = split_input_params
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False
        self.movie_len = movie_len
        self.downsample_image_size = downsample_img_size
        if unet_config['params']['ckpt_path'] is not None:
            # self.model.from_pretrained_model(unet_config['params']['ckpt_path'])
            self.from_pretrained_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'])
            self.restarted_from_ckpt = True

    def from_pretrained_model(self,model_path,ignore_keys):
        if not os.path.exists(model_path):
            raise RuntimeError(f'{model_path} does not exist')
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        my_model_state_dict = self.state_dict()
        keys = list(sd.keys())
        # for param in state_dict.keys():
        #     if param not in my_model_state_dict.keys():
        #         print("Missing Key:"+str(param))
        #     else:
        #         for ignore_key in ignore_keys:
        #             if not param.startswith(ignore_key):
        #                 params[param] = state_dict[param]
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
        #torch.save(my_model_state_dict,'./my_model_state_dict.pt')
        self.load_state_dict(sd,strict=False)

    def make_cond_schedule(self,):
        self.cond_ids = torch.full(size=(self.num_timesteps,),fill_value=self.num_timesteps - 1,dtype=torch.long)
        ids = torch.round(torch.linspace(0,self.num_timesteps - 1 ,self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids
    
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self,batch,batch_idx,current_epoch):
        #only for very first batch
        if self.scale_by_std and current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            #set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch,self.first_stage_key)
            encoder_posterior = self.encode_first_stage(x,self.first_stage_model)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor',1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
            return
        

    def instantiate_cond_stage(self,config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.clip = model.eval()
            self.clip.train = disabled_train
            for param in self.clip.parameters():
                param.requires_grad = False
            
        else:
            model = instantiate_from_config(config)
            self.clip = model

    def _get_denoise_row_from_list(self,samples,desc='',force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples,desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row) #n_log_step n_row C H W
        denoise_grid = rearrange(denoise_row,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_row)
        return denoise_grid
        
    def get_first_stage_encoding(self,encoder_posterior):
        if isinstance(encoder_posterior,DiagonalGuassianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior,torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def get_learned_conditioning(self,c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model,'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c,DiagonalGuassianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model,self.cond_stage_forward)
            c = getattr(self.cond_stage_model,self.cond_stage_forward)(c)
        return c

    def meshgrid(self,h,w):
        y = torch.arange(0,h).view(h,1,1).repeat(1,w,1)
        x = torch.arange(0,w).view(1,w,1).repeat(h,1,1)
        arr = torch.cat([y,x],dim=-1)
        return arr
    
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

    def training_step(self,batch,batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
    
    # def get_first_stage_encoding(self,encoder_posterior):
    #     if isinstance(encoder_posterior,DiagonalGuassianDistribution):
    #         z = encoder_posterior.sample()
    #     elif isinstance(encoder_posterior,torch.Tensor):
    #         z = encoder_posterior
    #     else:
    #         raise NotImplementedError(f'encoder_posterior of type {type(encoder_posterior)} not yet implemented')
    #     return self.scale_factor * z

    #check whether need / 255.
    @torch.no_grad()
    def encode_first_stage(self,x,encoder):
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
                return encoder.encode(x)
        else:
            return encoder.encode(x)
        

    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False):
        ref_img = super().get_input(batch,'reference_image')
        hdmap = super().get_input(batch,'HDmap')
        boxes = batch['3Dbox']
        boxes_category = batch['category']
        text = batch['text']
        boxes = rearrange(boxes,'b n x c -> (b n) x c')
        boxes_category = rearrange(boxes_category,'b n x c -> (b n) x c')
        text = rearrange(text,'b n c -> (b n) c').unsqueeze(1)
        range_image = batch['range_image']
        range_image = rearrange(range_image,'b n h w c -> (b n) c h w')
        depth_cam_front_img = batch['depth_cam_front_img']
        depth_cam_front_img = rearrange(depth_cam_front_img,'b n h w c -> (b n) c h w ')

        x = ref_img
        z = self.encode_first_stage(x,self.first_stage_model)
        # z = self.first_stage_model.encode(x)
        z = self.get_first_stage_encoding(z)
        lidar_z = self.get_first_stage_encoding(self.encode_first_stage(range_image,self.range_image_encoder))
        hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
        depth_cam_front_img = self.get_first_stage_encoding(self.encode_first_stage(depth_cam_front_img,self.range_image_encoder))
        out = [z,lidar_z]
        c = {}
        c['hdmap'] = hdmap.to(torch.float32)
        c['boxes'] = boxes.to(torch.float32)
        c['category'] = boxes_category.to(torch.float32)
        c['text'] = text.to(torch.float32)
        c['depth_cam_front_img'] = depth_cam_front_img.to(torch.float32)
        if return_first_stage_outputs:
            x_rec = self.decode_first_stage(z)
            lidar_rec = self.decode_first_stage(lidar_z)
            out.extend([x,x_rec,range_image,lidar_rec])
        out.append(c)
        return out
        # ref_img = super().get_input(batch,'reference_image')
        # hdmap = super().get_input(batch,'HDmap')
        # boxes = batch['3Dbox'].to(torch.float32).to(self.device)
        # boxes_category = batch['category']
        # text = batch['text']
        # x_embed = self.encode_first_stage(ref_img,self.ref_img_encoder)
        # hdmap_emb = self.encode_first_stage(hdmap,self.hdmap_encoder)
        # x_embed = self.get_first_stage_encoding(x_embed).detach()
        # hdmap_emb = self.get_first_stage_encoding(hdmap_emb).detach()
        
        # if self.use_positional_encodings:
        #     boxes_emb = rearrange(boxes_emb,'b n h w -> b (n h) w')
        #     # TODO:whether need add postional encoder
        #     # boxes_postional_encoder = PositionalEncoder(d_model=boxes_emb.shape[2],seq_len=boxes_emb.shape[1])
        #     # boxes_emb = boxes_postional_encoder(boxes_emb)
        #     # boxes_category_postional_encoder = PositionalEncoder(d_model=boxes_category.shape[2],seq_len=boxes_category.shape[1])
        #     # boxes_category = boxes_category_postional_encoder(boxes_category)
        #     boxes_emb = self.box_encoder.encode(boxes,boxes_category)
        # else:
        #     boxes_emb = boxes
        # c = {'hdmap_emb':hdmap_emb,
        #     'boxes_emb':boxes_emb,
        #      'box_clip':boxes_category,
        #      'text':text}
        # out = [x_embed,c]
        # if return_first_stage_outputs:
        #     xrec = self.decode_first_stage(x_embed)
        #     out.extend([ref_img,xrec])
        # return out
        
    # c = {'hdmap':...,"boxes":...,'category':...,"text":...}
    def shared_step(self,batch,**kwargs):
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        hdmap = super().get_input(batch,'HDmap') # (b n c h w)
        if 'range_image' in batch.keys():
            range_image = super().get_input(batch,'range_image')
            depth_cam_front_img = super().get_input(batch,'depth_cam_front_img')
        else:
            range_image = None
            depth_cam_front_img = None
        x = ref_img.to(torch.float32)
        x = self.get_first_stage_encoding(self.encode_first_stage(x,self.first_stage_model))
        c = {}
        hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
        if 'range_image' in batch.keys():
            range_image = self.get_first_stage_encoding(self.encode_first_stage(range_image,self.range_image_encoder))
            depth_cam_front_img = self.get_first_stage_encoding(self.encode_first_stage(depth_cam_front_img,self.range_image_encoder))
        c['hdmap'] = hdmap.to(torch.float32)
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d')
        boxes_category = rearrange(batch['category'],'b n c d -> (b n) c d')
        c['boxes'] = boxes.to(torch.float32)
        c['category'] = boxes_category.to(torch.float32)
        c['text'] = rearrange(batch['text'],'b n d -> (b n) d').unsqueeze(1).to(torch.float32)
        c['depth_cam_front_img'] = depth_cam_front_img
        loss,loss_dict = self(x,range_image,c)
        return loss,loss_dict

    def forward(self,x,range_image,c,*args,**kwargs):
        t = torch.randint(0,self.num_timesteps,(x.shape[0],),device=self.device).long()#
        return self.p_losses(x,range_image,c,t,*args,**kwargs)

    def apply_model(self,x_noisy,t,cond,range_image_noisy=None,x_start=None,return_ids=False):
        #TODO:wrong
        if hasattr(self,"split_input_params"):
            ks = self.split_input_params["ks_dif"]  # eg. (128, 128)
            stride = self.split_input_params["stride_dif"]  # eg. (64, 64)
            start_time = time.time()
            h, w = x_noisy.shape[-2:]
            fold,unfold,normalization,weighting = self.get_fold_unfold(x_noisy,ks,stride)

            z = unfold(x_noisy) # (bn, nc*prod(**ks),L)
            # Reshape to img shape
            z = z.view((z.shape[0],-1,ks[0],ks[1],z.shape[-1])) # (bn,nc,ks[0],ks[1],L)
            z_list = [z[:,:,:,:,i] for i in range(z.shape[-1])]

            #TODO:process condition
            #HDmap
            fold_hdmap,unfold_hdmap,normalization_hdmap,weighting_hdmap = self.get_fold_unfold(cond['hdmap'],ks,stride)
            hdmap_split = unfold_hdmap(cond['hdmap'])
            hdmap_split = hdmap_split.view((hdmap_split.shape[0],-1,ks[0],ks[1],hdmap_split.shape[-1]))
            hdmap_list = [hdmap_split[:,:,:,:,i] for i in range(hdmap_split.shape[-1])]

            #boxes
            boxes_list = []
            boxes_category_list = []
            full_img_w,full_img_h = self.split_input_params['original_image_size']
            col_h = np.arange(0,h-ks[0]+1,stride[0]) / x_noisy.shape[-2]
            row_w = np.arange(0,w-ks[1]+1,stride[1]) / x_noisy.shape[-1]
            boxes = cond['boxes'].reshape(-1,2,8)
            boxes[:,0] /= full_img_w
            boxes[:,1] /= full_img_h
            boxes = cond['boxes'].reshape(x_noisy.shape[0],-1,2,8)
            boxes_category = cond['category']
            ks_scale = [ks[0]/x_noisy.shape[-2],ks[1]/x_noisy.shape[-1]]
            for i in range(len(col_h)):
                for j in range(len(row_w)):
                    min_y = row_w[j]
                    min_x = col_h[i]
                    max_y = row_w[j] + ks_scale[1]
                    max_x = col_h[i] + ks_scale[0]
                    visible = np.logical_and(boxes[:,:,0,:].cpu()>=min_x,boxes[:,:,0,:].cpu()<=max_x)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()>=min_y)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()<=max_y)
                    visible = torch.any(visible,dim=-1,keepdim=True).bool()
                    boxes_mask = visible.unsqueeze(3).repeat(1,1,2,8).to(boxes.device)
                    patch_boxes = torch.masked_fill(boxes,boxes_mask,0).reshape(x_noisy.shape[0],-1,16)
                    boxes_list.append(patch_boxes)
                    category_mask = visible.repeat(1,1,768).to(boxes.device)
                    patch_category = torch.masked_fill(boxes_category,category_mask,0)
                    boxes_category_list.append(patch_category)

            output_list = []
            classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
            classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)
            boxes_emb_list = []
            for i in range(z.shape[-1]):
                z = z_list[i]
                # z = self.get_first_stage_encoding(self.encode_first_stage(z,self.first_stage_model))
                hdmap = hdmap_list[i]
                # hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
                z = torch.cat([z,hdmap],dim=1)
                boxes = boxes_list[i]
                
                #boxes = rearrange(boxes,'b n c -> (b n) c')
                boxes_category = boxes_category_list[i]
                boxes_emb = self.box_encoder(boxes,boxes_category)
                boxes_emb_list.append(boxes_emb)
                text_emb = cond['text']
                output = self.model(z,t,y=classes_emb[0],boxes_emb=boxes_emb,text_emb=text_emb)
                output_list.append(output)

            if not range_image_noisy is None:
                #range_image
                fold_range_image,unfold_range_image,normalization_range_image,weighting_range_image = self.get_fold_unfold(range_image_noisy,ks,stride)
                range_image_split = unfold_range_image(range_image_noisy)
                range_image_split = range_image_split.view((range_image_split.shape[0],-1,ks[0],ks[1],range_image_split.shape[-1]))
                range_image_split = [range_image_split[:,:,:,:,i] for i in range(range_image_split.shape[-1])]

                #depth_cam_front_img
                fold_depth_cam_front_img,unfold_depth_cam_front_img,normalization_depth_cam_front_img,weighting_depth_cam_front_img = self.get_fold_unfold(cond['depth_cam_front_img'],ks,stride)
                depth_cam_front_img_split = unfold_depth_cam_front_img(cond['depth_cam_front_img'])
                depth_cam_front_img_split = depth_cam_front_img_split.view(depth_cam_front_img_split.shape[0],-1,ks[0],ks[1],depth_cam_front_img_split.shape[-1])
                depth_cam_front_img_split = [depth_cam_front_img_split[:,:,:,:,i] for i in range(depth_cam_front_img_split.shape[-1])]

                #x_start
                fold_x_start,unfold_x_start,normalization_x_start,weighting_x_start = self.get_fold_unfold(x_start,ks,stride)
                x_start_split = unfold_x_start(x_start)
                x_start_split = x_start_split.view(x_start_split.shape[0],-1,ks[0],ks[1],x_start_split.shape[-1])
                x_start_split = [x_start_split[:,:,:,:,i] for i in range(x_start_split.shape[-1])]
            
                range_image_output = []
                for i in range(len(range_image_split)):
                    range_image = range_image_split[i]
                    x_start = x_start_split[i]
                    depth_cam_front_img = depth_cam_front_img_split[i]
                    text_emb = cond['text']
                    range_image = torch.cat((range_image,x_start),dim=1)
                    boxes_emb = boxes_emb_list[i]
                    output = self.model(range_image,t,y=classes_emb[1],boxes_emb=boxes_emb,text_emb=depth_cam_front_img.reshape(depth_cam_front_img.shape[0],1,-1))
                    range_image_output.append(output)
                o_ori = torch.stack(output_list,axis=-1)
                o_ori = o_ori * weighting
                # Reverse reshape to img shape
                o_ori = o_ori.view((o_ori.shape[0],-1,o_ori.shape[-1])) #(bn,nc*ks[0]*ks[1],L)
                # stitch crops together
                x_recon = fold(o_ori) / normalization
                o_range = torch.stack(range_image_output,axis=-1)
                o_range = o_range * weighting_range_image
                o_range = o_range.view((o_range.shape[0],-1,o_range.shape[-1]))
                range_recon = fold_range_image(o_range) / normalization_range_image
                return x_recon,range_recon
            return x_recon
        else:
            # hdmap = self.get_first_stage_encoding(self.encode_first_stage(cond['hdmap'],self.hdmap_encoder))
            boxes_emb = self.box_encoder(cond['boxes'],cond['category'])
            text_emb = cond['text']
            depth_cam_front_img = cond['depth_cam_front_img']
            classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
            classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)
            z = torch.cat([x_noisy,cond['hdmap']],dim=1)
            #TODO:need reshape z
            x_recon = self.model(z,t,y=classes_emb[0],boxes_emb=boxes_emb,text_emb=text_emb)
            z = torch.cat([range_image_noisy,x_start],dim=1)
            range_recon = self.model(z,t,y=classes_emb[1],boxes_emb=boxes_emb,text_emb=depth_cam_front_img.reshape(depth_cam_front_img.shape[0],4,-1))
        return x_recon,range_recon


    
    def p_losses(self,x_start,range_image_start,cond,t,noise=None):
        noise = default(noise,lambda:torch.randn_like(x_start)).to(x_start.device)        
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
        if not range_image_start is None:
            range_image_noisy = self.q_sample(x_start=range_image_start,t=t,noise=noise)
        else:
            range_image_noisy = None
        x_recon,lidar_recon = self.apply_model(x_noisy,t,cond,range_image_noisy,x_start)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == 'x0':
            target_x,target_lidar = x_start,range_image_start
        elif self.parameterization == 'eps':
            target_x,target_lidar = noise,noise
        else:
            raise NotImplementedError()
        if not range_image_start is None:
            loss_simple = self.get_loss(x_recon,target_x,mean=False).mean([1,2,3]) + self.get_loss(lidar_recon,target_lidar,mean=False).mean([1,2,3])
        else:
            loss_simple = self.get_loss(x_recon,target_x,mean=False).mean([1,2,3])
        loss_dict.update({f'{prefix}/loss_simple':loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma':loss.mean()})
            loss_dict.update({'logvar':self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if not range_image_start is None:
            loss_vlb = self.get_loss(x_recon,target_x,mean=False).mean(dim=[1,2,3]) + self.get_loss(lidar_recon,target_lidar,mean=False).mean([1,2,3])
        else:
            loss_vlb = self.get_loss(x_recon,target_x,mean=False).mean(dim=[1,2,3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb':loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss':loss})
        return loss,loss_dict
    
    def decode_first_stage(self,z,predict_cids=False,force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
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
            
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
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
                return self.first_stage_model.decode(z)
        else:
            if isinstance(self.first_stage_model,VQModelInterface):
                return self.first_stage_model.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
            
    def decode_first_stage_with_lidar(self,z,predict_cids=False,force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.range_image_encoder.quantize.get_codebook_entry(z, shape=None)
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
            
                output_list = [self.range_image_encoder.decode(z[:, :, :, :, i])
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
                return self.range_image_encoder.decode(z)
        else:
            if isinstance(self.range_image_encoder,VQModelInterface):
                return self.range_image_encoder.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.range_image_encoder.decode(z)
    
    def differentiable_decode_first_stage(self,z,predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1. / self.scale_factor * z

        if hasattr(self,"split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
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
            
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
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
                return self.first_stage_model.decode(z)
        else:
            if isinstance(self.first_stage_model,VQModelInterface):
                return self.first_stage_model.decode(z,force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
    
    def _rescale_annotations(self,bboxes,crop_coordinates):
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2],0,1)
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3],0,1)
            w = min(bbox[2] / crop_coordinates[2] , 1 - x0)
            h = min(bbox[3] / crop_coordinates[3] , 1 - y0)
            return x0,y0,w,h
        return [rescale_bbox(b) for b in bboxes]
    
    def _predict_eps_from_xstart(self,x_t,t,pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod,t,x_t.shape) * x_t - pred_xstart) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape) 
    
    def _prior_bpd(self,x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1]*batch_size,device=x_start.device)
        qt_mean,_,qt_log_variance = self.q_mean_variance(x_start,t)
        kl_prior = normal_kl(mean1=qt_mean,logvar1=qt_log_variance,
                             mean2=0.0,logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        if self.unet_trainable:
            print("add unet model parameters into optimizers")
            for name,param in self.model.named_parameters():
                # if 'temporal' in param or 'gated' in param:
                #     print(f"model:add {param} into optimizers")
                #     params.append(param)
                # if 'gated' in name or 'diffusion_model.input_blocks.0.0' in name or 'diffusion_model.out.2' in name:
                #     print(f"model:add {name} into optimizers")
                #     params.append(param)
                # elif 'transoformer'
                # elif not 'temporal' in name:
                #     param.requires_grad = False
                # else:
                #     # assert param.requires_grad == True
                #     param.requires_grad = False
                # if 'gated' in name or 'diffusion_model.input_blocks.0.0' in name or 'diffusion_model.out.2' in name:
                #     print(f"model:add {name} into optimizers")
                #     params.append(param)
                # elif not 'transformer_blocks' in name:
                #     param.requires_grad=False
                if not 'temporal' in name and 'diffusion_model' in name:
                    print(f"model:add {name} into optimizers")
                    params.append(param)
                
        if self.cond_stage_trainable:
            print("add encoder parameters into optimizers")
            params = params + list(self.box_encoder.parameters()) # + list(self.hdmap_encoder.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
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
        
    def p_mean_variance(self,x,c,t,clip_denoised:bool,return_codebook_ids=False,quantize_denoised=False,
                        return_x0=False,score_corrector=None,corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x,t_in,c,return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self,model_out,x,t,c,**corrector_kwargs)

        if return_codebook_ids:
            model_out,logits = model_out
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x,t=t,noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1.,1.)
        # if quantize_denoised:
        #     x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean,posterior_variance,posterior_log_variance = self.q_posterior(x_start=x_recon,x_t=x,t=t)
        if return_codebook_ids:
            return model_mean,posterior_variance,posterior_log_variance,logits
        elif return_x0:
            return model_mean,posterior_variance,posterior_log_variance,x_recon
        else:
            return model_mean,posterior_variance,posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self,x,c,t,clip_denoised=False,repeat_noise=False,
                 return_codebook_ids=False,quantize_denoised=False,return_x0=False,
                 temperature=1.,noise_dropout=0.,score_corrector=None,corrector_kwargs=None):
        b,*_,device = *x.shape,x.device
        outputs = self.p_mean_variance(x=x,c=c,t=t,clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean,_,model_log_variance,x0 = outputs
        else:
            model_mean,_,model_log_variance = outputs
        
        noise = noise_like(x.shape,device,repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise,p=noise_dropout)
        #no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
    @torch.no_grad()
    def progressive_denoising(self,cond,shape,verbose=True,callback=None,quantize_denoised=False,
                              img_callback=None,mask=None,x0=None,temperature=1.,noise_dropout=0.,
                              score_corrector=None,corrector_kwargs=None,batch_size=None,x_T=None,start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
                    range(0,timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        
        for i in iterator:
            ts = torch.full((b,),i,device=self.device,dtype=torch.long)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,t=tc,noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0,ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback:img_callback(img,i)
        return img,intermediates
    
    def p_sample_loop(self,cond,shape,return_intermediates=False,
                      x_T=None,verbose=True,callback=None,timesteps=None,quantize_denoised=False,
                      mask=None,x0=None,img_callback=None,start_T=None,log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=device)
        else:
            img = x_T
        
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        
        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Sampling t',total=timesteps) if verbose else reversed(
            range(0,timesteps))
        
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        
        for i in iterator:
            ts = torch.full((b,),i,device=device,dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if return_intermediates:
            return img,intermediates
        return img
    
    @torch.no_grad()
    def sample(self,cond,batch_size=16,return_intermediates=False,x_T=None,
               verbose=True,timesteps=None,quantize_denoised=False,
               mask=None,x0=None,shape=None,**kwargs):
        if shape is None:
            shape = (batch_size,self.channels) + tuple(self.image_size)
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]
        
        return self.p_sample_loop(cond,shape,return_intermediates=return_intermediates,
                                  x_T=x_T,verbose=verbose,timesteps=timesteps,
                                  quantize_denoised=quantize_denoised,mask=mask,x0=x0)
    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.downsample_image_size)
            samples,sample_lidar,intermediates = ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,sample_lidar,intermediates
    
    @torch.no_grad()
    def log_images(self,batch,N=8,n_row=4,sample=True,ddim_steps=200,ddim_eta=1.,return_keys=None,
                   quantize_denoised=True,inpaint=True,plot_denoise_rows=False,plot_progressive_rows=True,
                   plot_diffusion_rows=True,**kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z,lidar_z,x,x_rec,range_image,lidar_rec,c = self.get_input(batch,return_first_stage_outputs=True)

        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['inputs'] = x
        log['reconstruction'] = x_rec
        log['lidar_inputs'] = range_image
        log['lidar_reconstruction'] =  lidar_rec

        if plot_diffusion_rows:
            #get diffusion row
            diffusion_row = list()
            lidar_diffusion_row = list()
            z_start = z[:n_row]
            lidar_start = lidar_z[:n_row]
            
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]),'1 -> b',b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start,t=t,noise=noise)
                    lidar_noisy = self.q_sample(x_start=lidar_start,t=t,noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
                    lidar_diffusion_row.append(self.decode_first_stage(lidar_noisy))


            diffusion_row = torch.stack(diffusion_row) # n_log_step,n_row,C,H,W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log['diffusion_row'] = diffusion_grid
            lidar_diffusion_row = torch.stack(lidar_diffusion_row)
            lidar_diffusion_grid = rearrange(lidar_diffusion_row,'n b c h w -> b n c h w') 
            lidar_diffusion_grid = rearrange(lidar_diffusion_grid,'b n c h w -> (b n) c h w')
            lidar_diffusion_grid = make_grid(lidar_diffusion_grid,nrow=lidar_diffusion_row.shape[0])
            log['lidar_diffusion_row'] = lidar_diffusion_grid

        if sample:
            # get denoise row
            c['hdmap'] = c['hdmap'][:N]
            c['boxes'] = c['boxes'][:N]
            c['category'] = c['category'][:N]
            c['text'] = c['text'][:N]
            c['depth_cam_front_img'] = c['depth_cam_front_img'][:N]
            c['x_start'] = z[:N]
            with self.ema_scope("Plotting"):
                samples,sample_lidar,z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            lidar_sample = self.decode_first_stage_with_lidar(sample_lidar)
            log["samples"] = x_samples
            log['lidar_samples'] = lidar_sample
            # if plot_denoise_rows:
            #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            #     log['denoise_row'] = denoise_grid
            
            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
            #         self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
            #                                                  ddim_steps=ddim_steps,eta=ddim_eta,
            #                                                  quantize_denoised=True)
            #         # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
            #         #                                      quantize_denoised=True)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

            # if inpaint:
            #     #make a simple center square
            #     b,h,w = z.shape[0],z.shape[2],z.shape[3]
            #     mask = torch.ones(N,h,w).to(self.device)
            #     # zeros will be filled in
            #     mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            #     mask = mask[:, None, ...]
            #     with self.ema_scope("Plotting Inpaint"):

            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log['samples_inpainting'] = x_samples
            #     log["mask"] = mask

            #     #outpaint
            #     with self.ema_scope("Plotting Outpaint"):
            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_outpainting"] = x_samples
            # if plot_progressive_rows:
            #     with self.ema_scope("Plotting Progressives"):
            #         img, progressives = self.progressive_denoising(c,
            #                                                     shape=(self.channels, self.image_size[0],self.image_size[1]),
            #                                                     batch_size=N)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            # log["progressive_row"] = prog_row
            if return_keys:
                if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                    return log
                else:
                    return {key:log[key] for key in return_keys}
            return log
        
class AutoDM_GlobalCondition(DDPM):
    def __init__(self,
                 base_learning_rate,
                 global_condition_config,
                 num_timesteps_cond=None,
                 cond_stage_config=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unet_trainable=True,
                 split_input_params=None,
                 movie_len = 1,
                 predict=False,
                 downsample_img_size = None,
                 init_from_video_model=False,
                 actionformer_config=None,
                 use_additional_loss=False,
                 *args,**kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond,1)
        self.scale_by_std = scale_by_std
        self.learning_rate = base_learning_rate
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = 'concat' if concat_mode else 'crossattn'
        # if cond_stage_config == '__is_unconditional__':
        #     conditioning_key = None
        super().__init__(conditioning_key=None,*args,**kwargs)
        unet_config = kwargs.pop('unet_config',[])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.unet_trainable = unet_trainable
        global_condition_config['params']['learning_rate'] = self.learning_rate
        self.global_condition = instantiate_from_config(global_condition_config)
        self.use_additional_loss = use_additional_loss
        if predict:
            self.action_former = instantiate_from_config(actionformer_config)
        self.instantiate_cond_stage(cond_stage_config)
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor',torch.tensor(scale_factor))
        if not split_input_params is None:
            self.split_input_params = split_input_params
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False
        self.movie_len = movie_len
        self.downsample_image_size = downsample_img_size
        self.init_from_video_model = init_from_video_model
        self.predict = predict
        if unet_config['params']['ckpt_path'] is not None:
            # self.model.from_pretrained_model(unet_config['params']['ckpt_path'])
            if self.init_from_video_model:
                self.from_video_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'],unet_config['params']['modify_keys'])
            else:
                self.from_pretrained_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'])
            self.restarted_from_ckpt = True

    def from_video_model(self,model_path,ignore_keys,modify_keys):
        if not os.path.exists(model_path):
            raise RuntimeError(f"{model_path} does not exist")
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
            for ik in modify_keys:
                if k.startswith(ik):
                    modify_k = 'global_condition.' + k
                    sd[modify_k] = sd[k]
                    del(sd[k])
                    print("Modify Key {} to {} in state_dict".format(k,modify_k))
        self.load_state_dict(sd,strict=False)


    def from_pretrained_model(self,model_path,ignore_keys):
        if not os.path.exists(model_path):
            raise RuntimeError(f'{model_path} does not exist')
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        my_model_state_dict = self.state_dict()
        keys = list(sd.keys())
        # for param in state_dict.keys():
        #     if param not in my_model_state_dict.keys():
        #         print("Missing Key:"+str(param))
        #     else:
        #         for ignore_key in ignore_keys:
        #             if not param.startswith(ignore_key):
        #                 params[param] = state_dict[param]
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
        #torch.save(my_model_state_dict,'./my_model_state_dict.pt')
        self.load_state_dict(sd,strict=False)

    def make_cond_schedule(self,):
        self.cond_ids = torch.full(size=(self.num_timesteps,),fill_value=self.num_timesteps - 1,dtype=torch.long)
        ids = torch.round(torch.linspace(0,self.num_timesteps - 1 ,self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids
    
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self,batch,batch_idx,current_epoch):
        #only for very first batch
        if self.scale_by_std and current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            #set rescale weight to 1./std of encodings
            self.global_condition.get_scale_factor(batch)
        

    def instantiate_cond_stage(self,config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.clip = model.eval()
            self.clip.train = disabled_train
            for param in self.clip.parameters():
                param.requires_grad = False
            
        else:
            model = instantiate_from_config(config)
            self.clip = model

    def _get_denoise_row_from_list(self,samples,desc='',force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples,desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row) #n_log_step n_row C H W
        denoise_grid = rearrange(denoise_row,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_row)
        return denoise_grid
        
    def meshgrid(self,h,w):
        y = torch.arange(0,h).view(h,1,1).repeat(1,w,1)
        x = torch.arange(0,w).view(1,w,1).repeat(h,1,1)
        arr = torch.cat([y,x],dim=-1)
        return arr
    
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

    #TODO:add lidar_model loss
    def training_step(self,batch,batch_idx):

        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        # encoder_loss,encoder_loss_dict = self.global_condition.get_losses(batch)
        # loss = encoder_loss + loss 
        # self.log_dict(encoder_loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        # self.manual_backward(loss)
        # opt = self.optimizers()
        # opt.step()
        # opt.zero_grad()

        # encoder_loss,encoder_loss_dict = self.global_condition.get_losses(batch)
        # self.log_dict(encoder_loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        # self.manual_backward(encoder_loss)
        # if self.global_condition.image_model.trainable:
        #     opts = self.global_condition.image_model.optimizer
        #     for opt in opts:
        #         opt.step()
        #         opt.zero_grad()
        # if self.global_condition.lidar_model.trainable:
        #     opts = self.global_condition.lidar_model.optimizer
        #     for opt in opts:
        #         opt.step()
        #         opt.zero_grad()
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        # _, encoder_loss_dict = self.global_condition.get_losses(batch)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(encoder_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
    
    # def get_first_stage_encoding(self,encoder_posterior):
    #     if isinstance(encoder_posterior,DiagonalGuassianDistribution):
    #         z = encoder_posterior.sample()
    #     elif isinstance(encoder_posterior,torch.Tensor):
    #         z = encoder_posterior
    #     else:
    #         raise NotImplementedError(f'encoder_posterior of type {type(encoder_posterior)} not yet implemented')
    #     return self.scale_factor * z   

    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False):
        condition_keys = batch.keys()
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        batch['reference_image'] = ref_img
        if 'HDmap' in condition_keys:
            hdmap = super().get_input(batch,'HDmap') # (b n c h w)
            batch['HDmap'] = hdmap
        if 'range_image' in condition_keys:
            range_image = super().get_input(batch,'range_image') # (b n c h w)
            batch['range_image'] = range_image
        if 'dense_range_image' in condition_keys:
            dense_range_image = super().get_input(batch,'dense_range_image') # (b n c h w)
            batch['dense_range_image'] = dense_range_image
        if 'bev_images' in condition_keys:
            bev_images = super().get_input(batch,'bev_images')
            batch['bev_images'] = bev_images
        if '3Dbox' in condition_keys:
            boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()
            boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
            for i in range(len(batch['category'])):
                for j in range(self.movie_len):
                    for k in range(len(batch['category'][0][0])):
                        boxes_category[i][j][k] = self.clip(batch['category'][i][j][k]).cpu().detach().numpy()
            boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
            boxes_category = torch.tensor(boxes_category).to(self.device)
            batch['3Dbox'] = boxes
            batch['category'] = boxes_category
        if 'text' in condition_keys:
            text = np.zeros((len(batch['text']),self.movie_len,768))
            for i in range(len(batch['text'])):
                for j in range(self.movie_len):
                    text[i,j] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
            text = text.reshape(b*self.movie_len,1,768)
            text = torch.tensor(text).to(self.device)
            batch['text'] = text
        if 'actions' in condition_keys:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        condition_out = self.global_condition.get_conditions(batch)

        z = condition_out['ref_image']
        lidar_z = condition_out['range_image']
        
        out = [z,lidar_z]
        condition_keys = condition_out.keys()
        c = {}
        if 'hdmap' in condition_keys:
            c['hdmap'] = condition_out['hdmap'].to(torch.float32)
        if 'boxes_emb' in condition_keys:
            c['boxes_emb'] = condition_out['boxes_emb'].to(torch.float32)
        if 'text_emb' in condition_keys:
            c['text_emb'] = condition_out['text_emb'].to(torch.float32)
        if 'dense_range_image' in condition_keys:
            c['dense_range_image'] = condition_out['dense_range_image']
        if 'actions' in condition_keys:
            c['actions'] = condition_out['actions']
        if 'bev_images' in condition_keys:
            c['bev_images'] = condition_out['bev_images']
        if return_first_stage_outputs:
            x_rec = self.global_condition.decode_first_stage_interface("reference_image",z)
            lidar_rec = self.global_condition.decode_first_stage_interface('lidar',lidar_z)
            out.extend([batch['reference_image'],x_rec,batch['range_image'],lidar_rec])
        out.append(c)
        return out
        
    # c = {'hdmap':...,"boxes":...,'category':...,"text":...}
    def shared_step(self,batch,**kwargs):
        condition_keys = batch.keys()
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        batch['reference_image'] = ref_img
        if 'HDmap' in condition_keys:
            hdmap = super().get_input(batch,'HDmap') # (b n c h w)
            batch['HDmap'] = hdmap
        if 'range_image' in condition_keys:
            range_image = super().get_input(batch,'range_image') # (b n c h w)
            batch['range_image'] = range_image
        if 'dense_range_image' in condition_keys:
            dense_range_image = super().get_input(batch,'dense_range_image') # (b n c h w)
            batch['dense_range_image'] = dense_range_image
        if 'bev_images' in condition_keys:
            bev_images = super().get_input(batch,'bev_images')
            batch['bev_images'] = bev_images
        if '3Dbox' in condition_keys:
            boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()
            boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
            for i in range(len(batch['category'])):
                for j in range(self.movie_len):
                    for k in range(len(batch['category'][0][0])):
                        boxes_category[i][j][k] = self.clip(batch['category'][i][j][k]).cpu().detach().numpy()
            boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
            boxes_category = torch.tensor(boxes_category).to(self.device)
            batch['3Dbox'] = boxes
            batch['category'] = boxes_category
        if 'text' in condition_keys:
            text = np.zeros((len(batch['text']),self.movie_len,768))
            for i in range(len(batch['text'])):
                for j in range(self.movie_len):
                    text[i,j] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
            text = text.reshape(b*self.movie_len,1,768)
            text = torch.tensor(text).to(self.device)
            batch['text'] = text
        if 'actions' in condition_keys:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        out = self.global_condition.get_conditions(batch)
        condition_keys = out.keys()
        range_image = out['range_image']
        x = out['ref_image'].to(torch.float32)
        c = {}
        if 'hdmap' in condition_keys:
            c['hdmap'] = out['hdmap'].to(torch.float32)
        if 'boxes_emb' in condition_keys:
            c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        if 'text_emb' in condition_keys:
            c['text_emb'] = out['text_emb'].to(torch.float32)
        if 'dense_range_image' in condition_keys:
            c['dense_range_image'] = out['dense_range_image']
        if 'actions' in condition_keys:
            c['actions'] = out['actions']
        if 'bev_images' in condition_keys:
            c['bev_images'] = out['bev_images']
        c['origin_range_image'] = range_image
        loss,loss_dict = self(x,range_image,c)
        return loss,loss_dict

    def forward(self,x,range_image,c,*args,**kwargs):
        t = torch.randint(0,self.num_timesteps,(x.shape[0],),device=self.device).long()#
        return self.p_losses(x,range_image,c,t,*args,**kwargs)

    def apply_model(self,x_noisy,t,cond,range_image_noisy=None,x_start=None,return_ids=False):
        #TODO:wrong
        if hasattr(self,"split_input_params"):
            ks = self.split_input_params["ks_dif"]  # eg. (128, 128)
            stride = self.split_input_params["stride_dif"]  # eg. (64, 64)
            h, w = x_noisy.shape[-2:]
            fold,unfold,normalization,weighting = self.get_fold_unfold(x_noisy,ks,stride)

            z = unfold(x_noisy) # (bn, nc*prod(**ks),L)
            # Reshape to img shape
            z = z.view((z.shape[0],-1,ks[0],ks[1],z.shape[-1])) # (bn,nc,ks[0],ks[1],L)
            z_list = [z[:,:,:,:,i] for i in range(z.shape[-1])]

            #TODO:process condition
            #HDmap
            fold_hdmap,unfold_hdmap,normalization_hdmap,weighting_hdmap = self.get_fold_unfold(cond['hdmap'],ks,stride)
            hdmap_split = unfold_hdmap(cond['hdmap'])
            hdmap_split = hdmap_split.view((hdmap_split.shape[0],-1,ks[0],ks[1],hdmap_split.shape[-1]))
            hdmap_list = [hdmap_split[:,:,:,:,i] for i in range(hdmap_split.shape[-1])]

            #boxes
            boxes_list = []
            boxes_category_list = []
            full_img_w,full_img_h = self.split_input_params['original_image_size']
            col_h = np.arange(0,h-ks[0]+1,stride[0]) / x_noisy.shape[-2]
            row_w = np.arange(0,w-ks[1]+1,stride[1]) / x_noisy.shape[-1]
            boxes = cond['boxes'].reshape(-1,2,8)
            boxes[:,0] /= full_img_w
            boxes[:,1] /= full_img_h
            boxes = cond['boxes'].reshape(x_noisy.shape[0],-1,2,8)
            boxes_category = cond['category']
            ks_scale = [ks[0]/x_noisy.shape[-2],ks[1]/x_noisy.shape[-1]]
            for i in range(len(col_h)):
                for j in range(len(row_w)):
                    min_y = row_w[j]
                    min_x = col_h[i]
                    max_y = row_w[j] + ks_scale[1]
                    max_x = col_h[i] + ks_scale[0]
                    visible = np.logical_and(boxes[:,:,0,:].cpu()>=min_x,boxes[:,:,0,:].cpu()<=max_x)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()>=min_y)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()<=max_y)
                    visible = torch.any(visible,dim=-1,keepdim=True).bool()
                    boxes_mask = visible.unsqueeze(3).repeat(1,1,2,8).to(boxes.device)
                    patch_boxes = torch.masked_fill(boxes,boxes_mask,0).reshape(x_noisy.shape[0],-1,16)
                    boxes_list.append(patch_boxes)
                    category_mask = visible.repeat(1,1,768).to(boxes.device)
                    patch_category = torch.masked_fill(boxes_category,category_mask,0)
                    boxes_category_list.append(patch_category)

            output_list = []
            classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
            classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)
            boxes_emb_list = []
            for i in range(z.shape[-1]):
                z = z_list[i]
                # z = self.get_first_stage_encoding(self.encode_first_stage(z,self.first_stage_model))
                hdmap = hdmap_list[i]
                # hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
                z = torch.cat([z,hdmap],dim=1)
                boxes = boxes_list[i]
                
                #boxes = rearrange(boxes,'b n c -> (b n) c')
                boxes_category = boxes_category_list[i]
                boxes_emb = self.box_encoder(boxes,boxes_category)
                boxes_emb_list.append(boxes_emb)
                text_emb = cond['text']
                output = self.model(z,t,y=classes_emb[0],boxes_emb=boxes_emb,text_emb=text_emb)
                output_list.append(output)

            if not range_image_noisy is None:
                #range_image
                fold_range_image,unfold_range_image,normalization_range_image,weighting_range_image = self.get_fold_unfold(range_image_noisy,ks,stride)
                range_image_split = unfold_range_image(range_image_noisy)
                range_image_split = range_image_split.view((range_image_split.shape[0],-1,ks[0],ks[1],range_image_split.shape[-1]))
                range_image_split = [range_image_split[:,:,:,:,i] for i in range(range_image_split.shape[-1])]

                #depth_cam_front_img
                fold_depth_cam_front_img,unfold_depth_cam_front_img,normalization_depth_cam_front_img,weighting_depth_cam_front_img = self.get_fold_unfold(cond['depth_cam_front_img'],ks,stride)
                depth_cam_front_img_split = unfold_depth_cam_front_img(cond['depth_cam_front_img'])
                depth_cam_front_img_split = depth_cam_front_img_split.view(depth_cam_front_img_split.shape[0],-1,ks[0],ks[1],depth_cam_front_img_split.shape[-1])
                depth_cam_front_img_split = [depth_cam_front_img_split[:,:,:,:,i] for i in range(depth_cam_front_img_split.shape[-1])]

                #x_start
                fold_x_start,unfold_x_start,normalization_x_start,weighting_x_start = self.get_fold_unfold(x_start,ks,stride)
                x_start_split = unfold_x_start(x_start)
                x_start_split = x_start_split.view(x_start_split.shape[0],-1,ks[0],ks[1],x_start_split.shape[-1])
                x_start_split = [x_start_split[:,:,:,:,i] for i in range(x_start_split.shape[-1])]
            
                range_image_output = []
                for i in range(len(range_image_split)):
                    range_image = range_image_split[i]
                    x_start = x_start_split[i]
                    depth_cam_front_img = depth_cam_front_img_split[i]
                    text_emb = cond['text']
                    range_image = torch.cat((range_image,x_start),dim=1)
                    boxes_emb = boxes_emb_list[i]
                    output = self.model(range_image,t,y=classes_emb[1],boxes_emb=boxes_emb,text_emb=depth_cam_front_img.reshape(depth_cam_front_img.shape[0],1,-1))
                    range_image_output.append(output)
                o_ori = torch.stack(output_list,axis=-1)
                o_ori = o_ori * weighting
                # Reverse reshape to img shape
                o_ori = o_ori.view((o_ori.shape[0],-1,o_ori.shape[-1])) #(bn,nc*ks[0]*ks[1],L)
                # stitch crops together
                x_recon = fold(o_ori) / normalization
                o_range = torch.stack(range_image_output,axis=-1)
                o_range = o_range * weighting_range_image
                o_range = o_range.view((o_range.shape[0],-1,o_range.shape[-1]))
                range_recon = fold_range_image(o_range) / normalization_range_image
                return x_recon,range_recon
            return x_recon
        else:
            condition_keys = cond.keys()
            if 'bev_images' in condition_keys:
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                bev_images = cond['bev_images']
                z = torch.cat([x_noisy,bev_images],dim=1)
                lidar_z = torch.cat([range_image_noisy,bev_images],dim=1)
                actions = rearrange(cond['actions'],'b n c -> (b n) c')
                actions = torch.cat([actions,actions],dim=0)
                z = torch.cat([z,lidar_z],dim=0)
                classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
                classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)

                class_label = torch.zeros((z.shape[0],4),device=self.device)
                for i in range(z.shape[0]):
                    if i < z.shape[0] // 2:
                        class_label[i] = classes_emb[0]
                    else:
                        class_label[i] = classes_emb[1]
                #TODO:need reshape z
                x_recon = self.model(z,t,y=class_label,text_emb=text_emb,actions=actions)
            elif not 'action' in condition_keys:
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = torch.cat([boxes_emb,boxes_emb],dim=0)
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                if 'hdmap' in condition_keys and 'dense_range_image' in condition_keys:
                    dense_range_image = cond['dense_range_image']
                    z = torch.cat([x_noisy,cond['hdmap']],dim=1)
                    lidar_z = torch.cat([range_image_noisy,dense_range_image],dim=1)
                elif not 'hdmap' in condition_keys and not 'dense_range_image' in condition_keys:
                    z = x_noisy
                    lidar_z = range_image_noisy
                else:
                    raise NotImplementedError
                z = torch.cat([z,lidar_z])
                classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
                classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)

                class_label = torch.zeros((z.shape[0],4),device=self.device)

                for i in range(z.shape[0]):
                    if i < z.shape[0] // 2:
                        class_label[i] = classes_emb[0]
                    else:
                        class_label[i] = classes_emb[1]
                #TODO:need reshape z
                x_recon = self.model(z,t,y=class_label,boxes_emb=boxes_emb,text_emb=text_emb)
            else:
                b = x_noisy.shape[0] // self.movie_len
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = boxes_emb.reshape((b,self.movie_len)+boxes_emb.shape[1:])
                    boxes_emb = boxes_emb[:,0]
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                if 'hdmap' in condition_keys and 'dense_range_image' in condition_keys:
                    dense_range_image = cond['dense_range_image']
                    hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                    actions = cond['actions']
                    dense_range_image = dense_range_image.reshape((b,self.movie_len)+dense_range_image.shape[1:])[:,0]
                    h = self.action_former(hdmap,boxes_emb,actions,dense_range_image)
                    latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                    latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                    latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
                    boxes_emb = torch.cat([latent_boxes,latent_boxes],dim=0)

                    z = torch.cat([x_noisy,latent_hdmap],dim=1)
                    lidar_z = torch.cat([range_image_noisy,latent_dense_range_image],dim=1)
                    actions = None
                elif not 'hdmap' in condition_keys and not 'dense_range_image' in condition_keys:
                    z = x_noisy
                    lidar_z = range_image_noisy
                    actions = rearrange(cond['actions'],'b n c -> (b n) c')
                    actions = torch.cat([actions,actions],dim=0)
                else:
                    raise NotImplementedError
                z = torch.cat([z,lidar_z],dim=0)

                classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
                classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)

                class_label = torch.zeros((z.shape[0],4),device=self.device)
                for i in range(z.shape[0]):
                    if i < z.shape[0] // 2:
                        class_label[i] = classes_emb[0]
                    else:
                        class_label[i] = classes_emb[1]
                #TODO:need reshape z
                x_recon = self.model(z,t,y=class_label,boxes_emb=boxes_emb,text_emb=text_emb,actions=actions)



        return x_recon


    
    def p_losses(self,x_start,range_image_start,cond,t,noise=None):
        noise = default(noise,lambda:torch.randn_like(x_start)).to(x_start.device)        
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise)
        # x_noisy[::self.movie_len] = x_start[::self.movie_len]
        range_image_noisy = self.q_sample(x_start=range_image_start,t=t,noise=noise)
        # range_image_noisy[::self.movie_len] = range_image_start[::self.movie_len]
        t = torch.cat([t,t],dim=0)
        x_recon = self.apply_model(x_noisy,t,cond,range_image_noisy,x_start)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == 'x0':
            target_x = torch.cat([x_start,cond['origin_range_image']],dim=0)
        elif self.parameterization == 'eps':
            target_x = torch.cat([noise,noise],dim=0)
        else:
            raise NotImplementedError()
        if not range_image_start is None:
            loss_simple = self.get_loss(x_recon,target_x,mean=False).mean([1,2,3])
        else:
            loss_simple = self.get_loss(x_recon,target_x,mean=False).mean([1,2,3])
        loss_dict.update({f'{prefix}/loss_simple':loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma':loss.mean()})
            loss_dict.update({'logvar':self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(x_recon,target_x,mean=False).mean(dim=[1,2,3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb':loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss':loss})
        if self.use_additional_loss:
            x_rec = rearrange(x_recon,'(b t) c h w -> b t c h w',t=self.movie_len)
            inputs = torch.cat([x_noisy,range_image_noisy],dim=0)
            inputs = rearrange(inputs,'(b t) c h w -> b t c h w',t=self.movie_len)
            aux_loss = ((inputs[:,1:] - inputs[:,:-1])) - (x_rec)
            
        return loss,loss_dict
    
    
    def _rescale_annotations(self,bboxes,crop_coordinates):
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2],0,1)
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3],0,1)
            w = min(bbox[2] / crop_coordinates[2] , 1 - x0)
            h = min(bbox[3] / crop_coordinates[3] , 1 - y0)
            return x0,y0,w,h
        return [rescale_bbox(b) for b in bboxes]
    
    def _predict_eps_from_xstart(self,x_t,t,pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod,t,x_t.shape) * x_t - pred_xstart) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape) 
    
    def _prior_bpd(self,x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1]*batch_size,device=x_start.device)
        qt_mean,_,qt_log_variance = self.q_mean_variance(x_start,t)
        kl_prior = normal_kl(mean1=qt_mean,logvar1=qt_log_variance,
                             mean2=0.0,logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        if self.unet_trainable:
            print("add unet model parameters into optimizers")
            for name,param in self.model.named_parameters():
                if 'diffusion_model' in name:
                    print(f"model:add {name} into optimizers")
                    params.append(param)
                
        if self.cond_stage_trainable:
            print("add encoder parameters into optimizers")
            # param_names = self.global_condition.get_parameters_name()
            # for name,param in self.named_parameters():
            #     if name in param_names:
            #         print(f"add:{name}")
            #         params.append(param)
            params = params + self.global_condition.get_parameters()
        if self.predict:
            print("add action former parameters")
            params = params + list(self.action_former.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params,lr=lr)
        # opt = torch.optim.SGD(params,lr=lr)
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
        
    def p_mean_variance(self,x,c,t,clip_denoised:bool,return_codebook_ids=False,quantize_denoised=False,
                        return_x0=False,score_corrector=None,corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x,t_in,c,return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self,model_out,x,t,c,**corrector_kwargs)

        if return_codebook_ids:
            model_out,logits = model_out
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x,t=t,noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1.,1.)
        # if quantize_denoised:
        #     x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean,posterior_variance,posterior_log_variance = self.q_posterior(x_start=x_recon,x_t=x,t=t)
        if return_codebook_ids:
            return model_mean,posterior_variance,posterior_log_variance,logits
        elif return_x0:
            return model_mean,posterior_variance,posterior_log_variance,x_recon
        else:
            return model_mean,posterior_variance,posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self,x,c,t,clip_denoised=False,repeat_noise=False,
                 return_codebook_ids=False,quantize_denoised=False,return_x0=False,
                 temperature=1.,noise_dropout=0.,score_corrector=None,corrector_kwargs=None):
        b,*_,device = *x.shape,x.device
        outputs = self.p_mean_variance(x=x,c=c,t=t,clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean,_,model_log_variance,x0 = outputs
        else:
            model_mean,_,model_log_variance = outputs
        
        noise = noise_like(x.shape,device,repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise,p=noise_dropout)
        #no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
    @torch.no_grad()
    def progressive_denoising(self,cond,shape,verbose=True,callback=None,quantize_denoised=False,
                              img_callback=None,mask=None,x0=None,temperature=1.,noise_dropout=0.,
                              score_corrector=None,corrector_kwargs=None,batch_size=None,x_T=None,start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
                    range(0,timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        
        for i in iterator:
            ts = torch.full((b,),i,device=self.device,dtype=torch.long)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,t=tc,noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0,ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback:img_callback(img,i)
        return img,intermediates
    
    def p_sample_loop(self,cond,shape,return_intermediates=False,
                      x_T=None,verbose=True,callback=None,timesteps=None,quantize_denoised=False,
                      mask=None,x0=None,img_callback=None,start_T=None,log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=device)
        else:
            img = x_T
        
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        
        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Sampling t',total=timesteps) if verbose else reversed(
            range(0,timesteps))
        
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        
        for i in iterator:
            ts = torch.full((b,),i,device=device,dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if return_intermediates:
            return img,intermediates
        return img
    
    @torch.no_grad()
    def sample(self,cond,batch_size=16,return_intermediates=False,x_T=None,
               verbose=True,timesteps=None,quantize_denoised=False,
               mask=None,x0=None,shape=None,**kwargs):
        if shape is None:
            shape = (batch_size,self.channels) + tuple(self.image_size)
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]
        
        return self.p_sample_loop(cond,shape,return_intermediates=return_intermediates,
                                  x_T=x_T,verbose=verbose,timesteps=timesteps,
                                  quantize_denoised=quantize_denoised,mask=mask,x0=x0)
    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.downsample_image_size)
            samples,intermediates = ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,intermediates
    
    @torch.no_grad()
    def sample_video(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.image_size)
            samples = ddim_sampler.sample_video(ddim_steps,batch_size,shape,cond,verbose=False,movie_len=self.movie_len,**kwargs)
            return samples
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample_video_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.image_size)
            samples,intermediates = ddim_sampler.sample(ddim_steps,batch_size*self.movie_len,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,intermediates
    
    @torch.no_grad()
    def log_images(self,batch,N=8,n_row=4,sample=True,ddim_steps=200,ddim_eta=1.,return_keys=None,
                   quantize_denoised=True,inpaint=True,plot_denoise_rows=False,plot_progressive_rows=True,
                   plot_diffusion_rows=True,**kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z,lidar_z,x,x_rec,range_image,lidar_rec,c = self.get_input(batch,return_first_stage_outputs=True)

        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['inputs'] = x
        log['reconstruction'] = x_rec
        log['lidar_inputs'] = range_image
        log['lidar_reconstruction'] =  lidar_rec
        log['dense_range_image'] = c['dense_range_image']

        if plot_diffusion_rows:
            #get diffusion row
            diffusion_row = list()
            lidar_diffusion_row = list()
            z_start = z[:n_row]
            lidar_start = lidar_z[:n_row]
            
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]),'1 -> b',b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start,t=t,noise=noise)
                    lidar_noisy = self.q_sample(x_start=lidar_start,t=t,noise=noise)
                    diffusion_row.append(self.global_condition.decode_first_stage_interface('reference_image',z_noisy))
                    lidar_diffusion_row.append(self.global_condition.decode_first_stage_interface('lidar',lidar_noisy))


            diffusion_row = torch.stack(diffusion_row) # n_log_step,n_row,C,H,W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log['diffusion_row'] = diffusion_grid
            lidar_diffusion_row = torch.stack(lidar_diffusion_row)
            lidar_diffusion_grid = rearrange(lidar_diffusion_row,'n b c h w -> b n c h w') 
            lidar_diffusion_grid = rearrange(lidar_diffusion_grid,'b n c h w -> (b n) c h w')
            lidar_diffusion_grid = make_grid(lidar_diffusion_grid,nrow=lidar_diffusion_row.shape[0])
            log['lidar_diffusion_row'] = lidar_diffusion_grid

        if sample:
            # get denoise row
            c['hdmap'] = c['hdmap'][:N]
            c['boxes_emb'] = c['boxes_emb'][:N]
            c['text_emb'] = c['text_emb'][:N]
            c['dense_range_image'] = c['dense_range_image'][:N]
            c['x_start'] = z[:N]
            with self.ema_scope("Plotting"):
                samples,z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            samples,sample_lidar = torch.chunk(samples,2,dim=0)
            x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
            lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
            log["samples"] = x_samples
            log['lidar_samples'] = lidar_sample
            # if plot_denoise_rows:
            #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            #     log['denoise_row'] = denoise_grid
            
            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
            #         self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
            #                                                  ddim_steps=ddim_steps,eta=ddim_eta,
            #                                                  quantize_denoised=True)
            #         # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
            #         #                                      quantize_denoised=True)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

            # if inpaint:
            #     #make a simple center square
            #     b,h,w = z.shape[0],z.shape[2],z.shape[3]
            #     mask = torch.ones(N,h,w).to(self.device)
            #     # zeros will be filled in
            #     mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            #     mask = mask[:, None, ...]
            #     with self.ema_scope("Plotting Inpaint"):

            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log['samples_inpainting'] = x_samples
            #     log["mask"] = mask

            #     #outpaint
            #     with self.ema_scope("Plotting Outpaint"):
            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_outpainting"] = x_samples
            # if plot_progressive_rows:
            #     with self.ema_scope("Plotting Progressives"):
            #         img, progressives = self.progressive_denoising(c,
            #                                                     shape=(self.channels, self.image_size[0],self.image_size[1]),
            #                                                     batch_size=N)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            # log["progressive_row"] = prog_row
            if return_keys:
                if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                    return log
                else:
                    return {key:log[key] for key in return_keys}
            return log
    
    def log_video(self,batch,N=8,n_row=4,sample=True,ddim_steps=200,ddim_eta=1.,return_keys=None,
                   quantize_denoised=True,inpaint=True,plot_denoise_rows=False,plot_progressive_rows=True,
                   plot_diffusion_rows=True,**kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        x = batch['reference_image']
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        batch = {k:v[:N].to(self.global_condition.device) if isinstance(v,torch.Tensor) else v[:N] for k,v in batch.items()}
        z,lidar_z,x,x_rec,range_image,range_image_rec,c = self.get_input(batch,return_first_stage_outputs=True)

        log['inputs'] = x.reshape((N,-1)+x.shape[1:])
        log['reconstruction'] = x_rec.reshape((N,-1)+x_rec.shape[1:])
        log["lidar_inputs"] = range_image.reshape((N,-1)+range_image.shape[1:])
        log['lidar_reconstruction'] = range_image_rec.reshape((N,-1)+range_image_rec.shape[1:])
        # log['dense_range_image'] = c['dense_range_image'].reshape((N,-1),c['dense_range_image'].shape[1:])
        if sample:
            condition_key = c.keys()
            if 'hdmap' in condition_key:
                c['hdmap'] = c['hdmap'][:N*self.movie_len]
            if 'boxes_emb' in condition_key:
                c['boxes_emb'] = c['boxes_emb'][:N*self.movie_len]
            if 'text_emb' in condition_key:
                c['text_emb'] = c['text_emb'][:N*self.movie_len]
            if 'dense_range_image' in condition_key:
                c['dense_range_image'] = c['dense_range_image'][:N*self.movie_len]
            c['x_start'] = z[:N*self.movie_len]
            with self.ema_scope("Plotting"):
                # batch = {k:v.to(self.model.device) for k,v in batch.items()}
                samples,z_denoise_row = self.sample_video_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            samples = samples.to(self.global_condition.device)
            samples,sample_lidar = torch.chunk(samples,2,dim=0)
            x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
            lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
            print(f"lidar_samples:{lidar_sample.shape}")
            print(f"x_samples:{x_samples.shape}")
            log["samples"] = x_samples.reshape((N,-1)+x_samples.shape[1:])
            log['lidar_samples'] = lidar_sample.reshape((N,-1)+lidar_sample.shape[1:])
        return log

    def get_infer_cond(self,batch):
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b c h w)
        hdmap = super().get_input(batch,'HDmap')# (b c h w)
        range_image = super().get_input(batch,'range_image') # (b c h w)
        dense_range_image = super().get_input(batch,'dense_range_image') # (b c h w)
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

        condition_out = self.global_condition.get_conditions(batch,return_encoder_posterior=True)
        z = condition_out['ref_image']
        lidar_z = condition_out['range_image']
        
        out = [z,lidar_z]
        c = {}
        c['hdmap'] = condition_out['hdmap'].to(torch.float32)
        c['boxes_emb'] = condition_out['boxes_emb'].to(torch.float32)
        c['text_emb'] = condition_out['text_emb'].to(torch.float32)
        c['dense_range_image'] = condition_out['dense_range_image']
        if self.predict:
            c['actions'] = condition_out['actions']
        out.extend([batch['reference_image'],batch['range_image']])
        out.append(c)
        return out

    def single_infer(self,batch,ddim_steps=200,ddim_eta=1.):
        use_ddim = ddim_steps is not None
        b = batch['reference_image'].shape[0]
        z,lidar_z,x_start,range_image_start,c = self.get_infer_cond(batch)
        log = {}
        log['inputs'] = x_start.reshape((b,-1)+x_start.shape[1:])
        log['lidar_inputs'] = range_image_start.reshape((b,-1)+range_image_start.shape[1:])
        c['x_start'] = z
        c['lidar_start'] = lidar_z
        samples = self.sample_video(cond=c,batch_size=b,ddim=use_ddim,ddim_steps=ddim_steps,eta=ddim_eta)
        samples,sample_lidar = torch.chunk(samples,2,dim=0)
        x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
        lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
        log['samples'] = x_samples.reshape((b,-1)+x_samples.shape[1:])
        log['lidar_samples'] = lidar_sample.reshape((b,-1)+lidar_sample.shape[1:])
        return log


class AutoDM_GlobalCondition2(DDPM):
    def __init__(self,
                 base_learning_rate,
                 global_condition_config,
                 num_timesteps_cond=None,
                 cond_stage_config=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unet_trainable=True,
                 split_input_params=None,
                 movie_len = 1,
                 predict=False,
                 downsample_img_size = None,
                 init_from_video_model=False,
                 actionformer_config=None,
                 use_additional_loss=False,
                 sampler_config=None,
                 denoiser_config=None,
                 loss_fn_config=None,
                 replace_cond_frames=False,
                 fixed_cond_frames=None,
                 calc_decoder_loss=False,
                 use_similar="JS",
                 training_strategy='full',
                 load_from_ema=False,
                 data_type='mini',
                 *args,**kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond,1)
        self.scale_by_std = scale_by_std
        self.learning_rate = base_learning_rate
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = 'concat' if concat_mode else 'crossattn'
        # if cond_stage_config == '__is_unconditional__':
        #     conditioning_key = None
        super().__init__(conditioning_key=None,*args,**kwargs)
        unet_config = kwargs.pop('unet_config',[])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.unet_trainable = unet_trainable
        global_condition_config['params']['learning_rate'] = self.learning_rate
        self.global_condition = instantiate_from_config(global_condition_config)
        self.use_additional_loss = use_additional_loss
        self.replace_cond_frames = replace_cond_frames
        if predict:
            self.action_former = instantiate_from_config(actionformer_config)
        self.instantiate_cond_stage(cond_stage_config)
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor',torch.tensor(scale_factor))
        if not split_input_params is None:
            self.split_input_params = split_input_params
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False
        self.movie_len = movie_len
        self.downsample_image_size = downsample_img_size
        self.init_from_video_model = init_from_video_model
        self.predict = predict
        self.training_strategy = training_strategy
        self.fixed_cond_frames = fixed_cond_frames
        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.loss_fn = instantiate_from_config(loss_fn_config)
        self.calc_decoder_loss = calc_decoder_loss
        self.use_similar = use_similar
        self.load_from_ema = load_from_ema
        if unet_config['params']['ckpt_path'] is not None:
            # self.model.from_pretrained_model(unet_config['params']['ckpt_path'])
            if self.init_from_video_model:
                self.from_video_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'],unet_config['params']['modify_keys'])
            else:
                self.from_pretrained_model(unet_config['params']['ckpt_path'],unet_config['params']['ignore_keys'])
            self.restarted_from_ckpt = True
        if 'safetensor_path' in unet_config['params'].keys() and unet_config['params']['safetensor_path'] is not None:
            self.restarted_from_ckpt = True
        if data_type=='mini':
            self.category_name = ['None','human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.wheelchair', 'human.pedestrian.stroller', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.construction_worker', 'animal', 'vehicle.car', 'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.trailer', 'movable_object.barrier', 'movable_object.trafficcone', 'movable_object.pushable_pullable', 'movable_object.debris', 'static_object.bicycle_rack']
        else:
            self.category_name = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface', 'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation', 'vehicle.ego','None']
        self.category = {}
        for name in self.category_name:
            self.category[name] = self.clip(name).cpu().detach().numpy()
        del self.clip

        self.clipped_text = {}

    def from_video_model(self,model_path,ignore_keys,modify_keys):
        if not os.path.exists(model_path):
            raise RuntimeError(f"{model_path} does not exist")
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
            for ik in modify_keys:
                if k.startswith(ik):
                    modify_k = 'global_condition.' + k
                    sd[modify_k] = sd[k]
                    del(sd[k])
                    print("Modify Key {} to {} in state_dict".format(k,modify_k))
        self.load_state_dict(sd,strict=False)


    def from_pretrained_model(self,model_path,ignore_keys):
        if not os.path.exists(model_path):
            raise RuntimeError(f'{model_path} does not exist')
        sd = torch.load(model_path,map_location='cpu')['state_dict']
        if self.load_from_ema:
            s_name2name = dict(zip(self.model_ema.m_name2s_name.values(),self.model_ema.m_name2s_name.keys()))
            for k in list(sd.keys()):
                if k.startswith('model_ema'):
                    v = sd[k]
                    k = k[len('model_ema.'):]
                    if k in s_name2name.keys():
                        k = 'model.' + s_name2name[k]
                        sd[k] = v
        keys = list(sd.keys())
        # for param in state_dict.keys():
        #     if param not in my_model_state_dict.keys():
        #         print("Missing Key:"+str(param))
        #     else:
        #         for ignore_key in ignore_keys:
        #             if not param.startswith(ignore_key):
        #                 params[param] = state_dict[param]
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Delete Key {} from state_dict'.format(k))
                    del(sd[k])
        #torch.save(my_model_state_dict,'./my_model_state_dict.pt')
        self.load_state_dict(sd,strict=False)

    def make_cond_schedule(self,):
        self.cond_ids = torch.full(size=(self.num_timesteps,),fill_value=self.num_timesteps - 1,dtype=torch.long)
        ids = torch.round(torch.linspace(0,self.num_timesteps - 1 ,self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids
    
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self,batch,batch_idx,current_epoch):
        #only for very first batch
        if self.scale_by_std and current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            #set rescale weight to 1./std of encodings
            self.global_condition.get_scale_factor(batch)
        

    def instantiate_cond_stage(self,config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.clip = model.eval()
            self.clip.train = disabled_train
            for param in self.clip.parameters():
                param.requires_grad = False
            
        else:
            model = instantiate_from_config(config)
            self.clip = model

    def _get_denoise_row_from_list(self,samples,desc='',force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples,desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row) #n_log_step n_row C H W
        denoise_grid = rearrange(denoise_row,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_row)
        return denoise_grid
        
    def meshgrid(self,h,w):
        y = torch.arange(0,h).view(h,1,1).repeat(1,w,1)
        x = torch.arange(0,w).view(1,w,1).repeat(h,1,1)
        arr = torch.cat([y,x],dim=-1)
        return arr
    
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

    #TODO:add lidar_model loss
    def training_step(self,batch,batch_idx,optimizer_idx=None):

        loss,loss_dict = self.shared_step(batch,optimizer_idx)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            opt = self.optimizers()
            if isinstance(opt,list):
                lr = opt[0].param_groups[0]['lr']
            else:
                lr = opt.param_groups[0]['lr']
            self.log('lr_abs',lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        # if self.calc_decoder_loss:
        #     decoder_loss,decoder_loss_dict = self.global_condition.calc_similarity_loss(batch)
        #     loss = decoder_loss + loss
        #     self.log_dict(decoder_loss_dict,prog_bar=True,logger=True,on_epoch=True)
        # encoder_loss,encoder_loss_dict = self.global_condition.get_losses(batch)

        # loss = encoder_loss + loss 
        # self.log_dict(encoder_loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)
        # self.manual_backward(loss)
        # opt = self.optimizers()
        # opt.step()
        # opt.zero_grad()

        # encoder_loss,encoder_loss_dict = self.global_condition.get_losses(batch)
        # self.log_dict(encoder_loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        # self.manual_backward(encoder_loss)
        # if self.global_condition.image_model.trainable:
        #     opts = self.global_condition.image_model.optimizer
        #     for opt in opts:
        #         opt.step()
        #         opt.zero_grad()
        # if self.global_condition.lidar_model.trainable:
        #     opts = self.global_condition.lidar_model.optimizer
        #     for opt in opts:
        #         opt.step()
        #         opt.zero_grad()
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        # _, encoder_loss_dict = self.global_condition.get_losses(batch)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(encoder_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False):
        condition_keys = batch.keys()
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        batch['reference_image'] = ref_img
        if 'HDmap' in condition_keys:
            hdmap = super().get_input(batch,'HDmap') # (b n c h w)
            batch['HDmap'] = hdmap
        if 'range_image' in condition_keys:
            range_image = super().get_input(batch,'range_image') # (b n c h w)
            batch['range_image'] = range_image
        if 'dense_range_image' in condition_keys:
            dense_range_image = super().get_input(batch,'dense_range_image') # (b n c h w)
            batch['dense_range_image'] = dense_range_image
        if 'bev_images' in condition_keys:
            bev_images = super().get_input(batch,'bev_images')
            batch['bev_images'] = bev_images
        if '3Dbox' in condition_keys:
            boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()
            boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
            for i in range(len(batch['category'])):
                for j in range(self.movie_len):
                    for k in range(len(batch['category'][0][0])):
                        boxes_category[i][j][k] = self.category[batch['category'][i][j][k]]
            boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
            boxes_category = torch.tensor(boxes_category).to(self.device)
            batch['3Dbox'] = boxes
            batch['category'] = boxes_category
        if 'text' in condition_keys:
            text = np.zeros((len(batch['text']),self.movie_len,768))
            for i in range(len(batch['text'])):
                for j in range(self.movie_len):
                    if not batch['text'][i][j] in self.clipped_text.keys():
                        self.clipped_text[batch['text'][i][j]] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
                    text[i,j] = self.clipped_text[batch['text'][i][j]]
            text = text.reshape(b*self.movie_len,1,768)
            text = torch.tensor(text).to(self.device)
            batch['text'] = text
        if 'actions' in condition_keys:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}

        condition_out = self.global_condition.get_conditions(batch)
        condition_keys = condition_out.keys()
        z = condition_out['ref_image']
        if 'range_image' in condition_keys:
            lidar_z = condition_out['range_image']
        else:
            lidar_z = None
        
        out = [z,lidar_z]
        c = {}
        if 'hdmap' in condition_keys:
            c['hdmap'] = condition_out['hdmap'].to(torch.float32)
        if 'boxes_emb' in condition_keys:
            c['boxes_emb'] = condition_out['boxes_emb'].to(torch.float32)
        if 'text_emb' in condition_keys:
            c['text_emb'] = condition_out['text_emb'].to(torch.float32)
        if 'dense_range_image' in condition_keys:
            c['dense_range_image'] = condition_out['dense_range_image']
        if 'actions' in condition_keys:
            c['actions'] = condition_out['actions']
        if 'bev_images' in condition_keys:
            c['bev_images'] = condition_out['bev_images']
        if 'cam_enc' in condition_keys:
            c['cam_enc'] = out['cam_enc']
        if 'lidar_enc' in condition_keys:
            c['lidar_enc'] = out['lidar_enc']
        c['origin_x'] = condition_out['ref_image'][::self.movie_len]
        c['origin_range'] = condition_out['range_image'][::self.movie_len]
        if return_first_stage_outputs:
            x_rec = self.global_condition.decode_first_stage_interface("reference_image",z)
            if lidar_z is None:
                out.extend([batch['reference_image'],x_rec,None,None])
            else:
                lidar_rec = self.global_condition.decode_first_stage_interface('lidar',lidar_z)
                out.extend([batch['reference_image'],x_rec,batch['range_image'],lidar_rec])
        out.append(c)
        return out
        
    # c = {'hdmap':...,"boxes":...,'category':...,"text":...}
    def shared_step(self,batch,optimizer_idx=None,**kwargs):
        condition_keys = batch.keys()
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b n c h w)
        batch['reference_image'] = ref_img
        if 'HDmap' in condition_keys:
            hdmap = super().get_input(batch,'HDmap') # (b n c h w)
            batch['HDmap'] = hdmap
        if 'range_image' in condition_keys:
            range_image = super().get_input(batch,'range_image') # (b n c h w)
            batch['range_image'] = range_image
        if 'dense_range_image' in condition_keys:
            dense_range_image = super().get_input(batch,'dense_range_image') # (b n c h w)
            batch['dense_range_image'] = dense_range_image
        if 'bev_images' in condition_keys:
            bev_images = super().get_input(batch,'bev_images')
            batch['bev_images'] = bev_images
        if '3Dbox' in condition_keys:
            boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d').contiguous()
            boxes_category = np.zeros((len(batch['category']),self.movie_len,len(batch['category'][0][0]),768))
            for i in range(len(batch['category'])):
                for j in range(self.movie_len):
                    for k in range(len(batch['category'][0][0])):
                        boxes_category[i][j][k] = self.category[batch['category'][i][j][k]]
            boxes_category = boxes_category.reshape(b*self.movie_len,len(batch['category'][0][0]),768)
            boxes_category = torch.tensor(boxes_category).to(self.device)
            batch['3Dbox'] = boxes
            batch['category'] = boxes_category
        if 'text' in condition_keys:
            text = np.zeros((len(batch['text']),self.movie_len,768))
            for i in range(len(batch['text'])):
                for j in range(self.movie_len):
                    if not batch['text'][i][j] in self.clipped_text.keys():
                        self.clipped_text[batch['text'][i][j]] = self.clip(batch['text'][i][j]).cpu().detach().numpy()
                    text[i,j] = self.clipped_text[batch['text'][i][j]]
            text = text.reshape(b*self.movie_len,1,768)
            text = torch.tensor(text).to(self.device)
            batch['text'] = text
        if 'actions' in condition_keys:
            batch['actions'] = batch['actions']
        batch = {k:v.to(torch.float32) for k,v in batch.items()}
        out = self.global_condition.get_conditions(batch,self.calc_decoder_loss)
        condition_keys = out.keys()
        if 'range_image' in out.keys():
            range_image = out['range_image']
        else:
            range_image = None
        x = out['ref_image'].to(torch.float32)
        c = {}
        if 'hdmap' in condition_keys:
            c['hdmap'] = out['hdmap'].to(torch.float32)
        if 'boxes_emb' in condition_keys:
            c['boxes_emb'] = out['boxes_emb'].to(torch.float32)
        if 'text_emb' in condition_keys:
            c['text_emb'] = out['text_emb'].to(torch.float32)
        if 'dense_range_image' in condition_keys:
            c['dense_range_image'] = out['dense_range_image']
        if 'actions' in condition_keys:
            c['actions'] = out['actions']
        if 'bev_images' in condition_keys:
            c['bev_images'] = out['bev_images']
        if 'cam_enc' in condition_keys:
            c['cam_enc'] = out['cam_enc']
        if 'lidar_enc' in condition_keys:
            c['lidar_enc'] = out['lidar_enc']
        c['origin_range_image'] = range_image
        c['origin_x'] = out['ref_image'][::self.movie_len]
        c['origin_range'] = out['range_image'][::self.movie_len]
        if self.calc_decoder_loss:
            x_rec,cam_dec = self.global_condition.decode_first_stage_interface("reference_image",x,calc_decoder_loss=self.calc_decoder_loss)
            lidar_rec,lidar_dec = self.global_condition.decode_first_stage_interface('lidar',range_image,calc_decoder_loss=self.calc_decoder_loss)
            c['cam_dec'] = cam_dec
            c['lidar_dec'] = lidar_dec
        if not hasattr(self.loss_fn,'discriminator'):
            loss,loss_dict = self(x,range_image,c)
        else:
            loss,loss_dict = self(x,range_image,c,optimizer_idx)
        return loss,loss_dict


    def forward(self,x,range_image,c,optimizer_idx=None,*args,**kwargs):
        if not optimizer_idx is None:
            if optimizer_idx == 0:
                loss,camera_loss,range_image_loss,g_loss = self.loss_fn(self.model,self.denoiser,c,x,range_image,self.action_former,self.global_step,self.calc_decoder_loss,self.use_similar,optimizer_idx,self.model.get_last_layer())
                log_prefix = "train" if self.training else 'val'
                loss_dict = {f"{log_prefix}/loss":camera_loss + range_image_loss,f"{log_prefix}/camera_loss":camera_loss,f"{log_prefix}/range_loss":range_image_loss,f"{log_prefix}/g_loss":g_loss}
            else:
                loss,logits_real,logits_fake = self.loss_fn(self.model,self.denoiser,c,x,range_image,self.action_former,self.global_step,self.calc_decoder_loss,self.use_similar,optimizer_idx)
                log_prefix = "train" if self.training else 'val'
                loss_dict = {f"{log_prefix}/d_loss":loss.clone().detach().mean(),f"{log_prefix}/logits_real":logits_real.detach().mean(),f"{log_prefix}/logits_fake":logits_fake.detach().mean()}
            return loss,loss_dict
        else:
            if self.calc_decoder_loss:
                loss,camera_loss,range_image_loss,simiarity = self.loss_fn(self.model,self.denoiser,c,x,range_image,self.action_former,self.global_step,self.calc_decoder_loss,self.use_similar)
                log_prefix = "train" if self.training else 'val'
                loss_dict = {f"{log_prefix}/loss":camera_loss + range_image_loss + simiarity,f"{log_prefix}/camera_loss":camera_loss,f"{log_prefix}/range_loss":range_image_loss,f"{log_prefix}/similarity":simiarity}
            else:    
                # loss,camera_loss,range_image_loss = self.loss_fn(self.model,self.denoiser,c,x,range_image,self.action_former,self.global_step,self.calc_decoder_loss,self.use_similar)
                # loss_mean = loss.mean()
                loss,camera_loss,range_image_loss = self.loss_fn(self.model,self.denoiser,c,x,range_image,self.action_former,self.global_step,self.calc_decoder_loss,self.use_similar)
                log_prefix = "train" if self.training else 'val'
                loss_dict = {f"{log_prefix}/loss":camera_loss + range_image_loss,f"{log_prefix}/camera_loss":camera_loss,f"{log_prefix}/range_loss":range_image_loss}
        return loss,loss_dict

    def apply_model(self,x_noisy,t,cond,range_image_noisy=None,x_start=None,return_ids=False):
        #TODO:wrong
        if hasattr(self,"split_input_params"):
            ks = self.split_input_params["ks_dif"]  # eg. (128, 128)
            stride = self.split_input_params["stride_dif"]  # eg. (64, 64)
            h, w = x_noisy.shape[-2:]
            fold,unfold,normalization,weighting = self.get_fold_unfold(x_noisy,ks,stride)

            z = unfold(x_noisy) # (bn, nc*prod(**ks),L)
            # Reshape to img shape
            z = z.view((z.shape[0],-1,ks[0],ks[1],z.shape[-1])) # (bn,nc,ks[0],ks[1],L)
            z_list = [z[:,:,:,:,i] for i in range(z.shape[-1])]

            #TODO:process condition
            #HDmap
            fold_hdmap,unfold_hdmap,normalization_hdmap,weighting_hdmap = self.get_fold_unfold(cond['hdmap'],ks,stride)
            hdmap_split = unfold_hdmap(cond['hdmap'])
            hdmap_split = hdmap_split.view((hdmap_split.shape[0],-1,ks[0],ks[1],hdmap_split.shape[-1]))
            hdmap_list = [hdmap_split[:,:,:,:,i] for i in range(hdmap_split.shape[-1])]

            #boxes
            boxes_list = []
            boxes_category_list = []
            full_img_w,full_img_h = self.split_input_params['original_image_size']
            col_h = np.arange(0,h-ks[0]+1,stride[0]) / x_noisy.shape[-2]
            row_w = np.arange(0,w-ks[1]+1,stride[1]) / x_noisy.shape[-1]
            boxes = cond['boxes'].reshape(-1,2,8)
            boxes[:,0] /= full_img_w
            boxes[:,1] /= full_img_h
            boxes = cond['boxes'].reshape(x_noisy.shape[0],-1,2,8)
            boxes_category = cond['category']
            ks_scale = [ks[0]/x_noisy.shape[-2],ks[1]/x_noisy.shape[-1]]
            for i in range(len(col_h)):
                for j in range(len(row_w)):
                    min_y = row_w[j]
                    min_x = col_h[i]
                    max_y = row_w[j] + ks_scale[1]
                    max_x = col_h[i] + ks_scale[0]
                    visible = np.logical_and(boxes[:,:,0,:].cpu()>=min_x,boxes[:,:,0,:].cpu()<=max_x)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()>=min_y)
                    visible = np.logical_and(visible,boxes[:,:,1,:].cpu()<=max_y)
                    visible = torch.any(visible,dim=-1,keepdim=True).bool()
                    boxes_mask = visible.unsqueeze(3).repeat(1,1,2,8).to(boxes.device)
                    patch_boxes = torch.masked_fill(boxes,boxes_mask,0).reshape(x_noisy.shape[0],-1,16)
                    boxes_list.append(patch_boxes)
                    category_mask = visible.repeat(1,1,768).to(boxes.device)
                    patch_category = torch.masked_fill(boxes_category,category_mask,0)
                    boxes_category_list.append(patch_category)

            output_list = []
            classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
            classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)
            boxes_emb_list = []
            for i in range(z.shape[-1]):
                z = z_list[i]
                # z = self.get_first_stage_encoding(self.encode_first_stage(z,self.first_stage_model))
                hdmap = hdmap_list[i]
                # hdmap = self.get_first_stage_encoding(self.encode_first_stage(hdmap,self.hdmap_encoder))
                z = torch.cat([z,hdmap],dim=1)
                boxes = boxes_list[i]
                
                #boxes = rearrange(boxes,'b n c -> (b n) c')
                boxes_category = boxes_category_list[i]
                boxes_emb = self.box_encoder(boxes,boxes_category)
                boxes_emb_list.append(boxes_emb)
                text_emb = cond['text']
                output = self.model(z,t,y=classes_emb[0],boxes_emb=boxes_emb,text_emb=text_emb)
                output_list.append(output)

            if not range_image_noisy is None:
                #range_image
                fold_range_image,unfold_range_image,normalization_range_image,weighting_range_image = self.get_fold_unfold(range_image_noisy,ks,stride)
                range_image_split = unfold_range_image(range_image_noisy)
                range_image_split = range_image_split.view((range_image_split.shape[0],-1,ks[0],ks[1],range_image_split.shape[-1]))
                range_image_split = [range_image_split[:,:,:,:,i] for i in range(range_image_split.shape[-1])]

                #depth_cam_front_img
                fold_depth_cam_front_img,unfold_depth_cam_front_img,normalization_depth_cam_front_img,weighting_depth_cam_front_img = self.get_fold_unfold(cond['depth_cam_front_img'],ks,stride)
                depth_cam_front_img_split = unfold_depth_cam_front_img(cond['depth_cam_front_img'])
                depth_cam_front_img_split = depth_cam_front_img_split.view(depth_cam_front_img_split.shape[0],-1,ks[0],ks[1],depth_cam_front_img_split.shape[-1])
                depth_cam_front_img_split = [depth_cam_front_img_split[:,:,:,:,i] for i in range(depth_cam_front_img_split.shape[-1])]

                #x_start
                fold_x_start,unfold_x_start,normalization_x_start,weighting_x_start = self.get_fold_unfold(x_start,ks,stride)
                x_start_split = unfold_x_start(x_start)
                x_start_split = x_start_split.view(x_start_split.shape[0],-1,ks[0],ks[1],x_start_split.shape[-1])
                x_start_split = [x_start_split[:,:,:,:,i] for i in range(x_start_split.shape[-1])]
            
                range_image_output = []
                for i in range(len(range_image_split)):
                    range_image = range_image_split[i]
                    x_start = x_start_split[i]
                    depth_cam_front_img = depth_cam_front_img_split[i]
                    text_emb = cond['text']
                    range_image = torch.cat((range_image,x_start),dim=1)
                    boxes_emb = boxes_emb_list[i]
                    output = self.model(range_image,t,y=classes_emb[1],boxes_emb=boxes_emb,text_emb=depth_cam_front_img.reshape(depth_cam_front_img.shape[0],1,-1))
                    range_image_output.append(output)
                o_ori = torch.stack(output_list,axis=-1)
                o_ori = o_ori * weighting
                # Reverse reshape to img shape
                o_ori = o_ori.view((o_ori.shape[0],-1,o_ori.shape[-1])) #(bn,nc*ks[0]*ks[1],L)
                # stitch crops together
                x_recon = fold(o_ori) / normalization
                o_range = torch.stack(range_image_output,axis=-1)
                o_range = o_range * weighting_range_image
                o_range = o_range.view((o_range.shape[0],-1,o_range.shape[-1]))
                range_recon = fold_range_image(o_range) / normalization_range_image
                return x_recon,range_recon
            return x_recon
        else:
            condition_keys = cond.keys()
            if not 'action' in condition_keys:
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = torch.cat([boxes_emb,boxes_emb],dim=0)
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                if 'hdmap' in condition_keys and 'dense_range_image' in condition_keys:
                    dense_range_image = cond['dense_range_image']
                    z = torch.cat([x_noisy,cond['hdmap']],dim=1)
                    lidar_z = torch.cat([range_image_noisy,dense_range_image],dim=1)
                elif not 'hdmap' in condition_keys and not 'dense_range_image' in condition_keys:
                    z = x_noisy
                    lidar_z = range_image_noisy
                else:
                    raise NotImplementedError
                z = torch.cat([z,lidar_z])
                classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
                classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)

                class_label = torch.zeros((z.shape[0],4),device=self.device)

                for i in range(z.shape[0]):
                    if i < z.shape[0] // 2:
                        class_label[i] = classes_emb[0]
                    else:
                        class_label[i] = classes_emb[1]
                #TODO:need reshape z
                
                if isinstance(self.model,UNetModel):
                    x_recon = self.model(z,t,y=class_label,boxes_emb=boxes_emb,text_emb=text_emb)
                elif isinstance(self.model,VideoUNet):
                    x_recon = self.model(z,t,y=class_label,context=boxes_emb,num_frames=self.movie_len)
            else:
                b = x_noisy.shape[0] // self.movie_len
                if 'boxes_emb' in condition_keys:
                    boxes_emb = cond['boxes_emb']
                    boxes_emb = boxes_emb.reshape((b,self.movie_len)+boxes_emb.shape[1:])
                    boxes_emb = boxes_emb[:,0]
                else:
                    boxes_emb = None
                if 'text_emb' in condition_keys:
                    text_emb = cond['text_emb']
                    text_emb = torch.cat([text_emb,text_emb],dim=0)
                else:
                    text_emb = None
                if 'hdmap' in condition_keys and 'dense_range_image' in condition_keys:
                    dense_range_image = cond['dense_range_image']
                    hdmap = cond['hdmap'].reshape((b,self.movie_len)+cond['hdmap'].shape[1:])[:,0]
                    actions = cond['actions']
                    dense_range_image = dense_range_image.reshape((b,self.movie_len)+dense_range_image.shape[1:])[:,0]
                    h = self.action_former(hdmap,boxes_emb,actions,dense_range_image)
                    latent_hdmap = torch.stack([h[id][0] for id in range(len(h))]).reshape(cond['hdmap'].shape)
                    latent_boxes = torch.stack([h[id][1] for id in range(len(h))]).reshape(cond['boxes_emb'].shape)
                    latent_dense_range_image = torch.stack([h[id][2] for id in range(len(h))]).reshape(cond['dense_range_image'].shape)
                    boxes_emb = torch.cat([latent_boxes,latent_boxes],dim=0)

                    z = torch.cat([x_noisy,latent_hdmap],dim=1)
                    lidar_z = torch.cat([range_image_noisy,latent_dense_range_image],dim=1)
                    actions = None
                elif not 'hdmap' in condition_keys and not 'dense_range_image' in condition_keys:
                    z = x_noisy
                    lidar_z = range_image_noisy
                    actions = rearrange(cond['actions'],'b n c -> (b n) c')
                    actions = torch.cat([actions,actions],dim=0)
                else:
                    raise NotImplementedError
                z = torch.cat([z,lidar_z],dim=0)

                classes = torch.tensor([[1.,0.],[0.,1.]],device=self.device,dtype=self.dtype)
                classes_emb = torch.cat([torch.sin(classes),torch.cos(classes)],dim=-1)

                class_label = torch.zeros((z.shape[0],4),device=self.device)
                for i in range(z.shape[0]):
                    if i < z.shape[0] // 2:
                        class_label[i] = classes_emb[0]
                    else:
                        class_label[i] = classes_emb[1]
                #TODO:need reshape z
                if isinstance(self.model,UNetModel):
                    x_recon = self.model(z,t,y=class_label,boxes_emb=boxes_emb,text_emb=text_emb,actions=actions)
                elif isinstance(self.model,VideoUNet):
                    x_recon = self.model(z,t,y=class_label,context=boxes_emb,num_frames=self.movie_len)


        return x_recon
    
    
    def _rescale_annotations(self,bboxes,crop_coordinates):
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2],0,1)
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3],0,1)
            w = min(bbox[2] / crop_coordinates[2] , 1 - x0)
            h = min(bbox[3] / crop_coordinates[3] , 1 - y0)
            return x0,y0,w,h
        return [rescale_bbox(b) for b in bboxes]
    
    def _predict_eps_from_xstart(self,x_t,t,pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod,t,x_t.shape) * x_t - pred_xstart) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape) 
    
    def _prior_bpd(self,x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1]*batch_size,device=x_start.device)
        qt_mean,_,qt_log_variance = self.q_mean_variance(x_start,t)
        kl_prior = normal_kl(mean1=qt_mean,logvar1=qt_log_variance,
                             mean2=0.0,logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        if self.training_strategy == 'full':
            if self.unet_trainable:
                print("add unet model parameters into optimizers")
                # for name,param in self.model.named_parameters():
                #     if 'diffusion_model' in name:
                #         print(f"model:add {name} into optimizers")
                #         params.append(param)
                params = params + list(self.model.parameters())
                    
            if self.cond_stage_trainable:
                print("add encoder parameters into optimizers")
                # param_names = self.global_condition.get_parameters_name()
                # for name,param in self.named_parameters():
                #     if name in param_names:
                #         print(f"add:{name}")
                #         params.append(param)
                params = params + self.global_condition.get_parameters()
            if self.predict:
                print("add action former parameters")
                params = params + list(self.action_former.parameters())
            if self.learn_logvar:
                print("Diffusion model optimizing logvar")
                params.append(self.logvar)
        # calc_decoder_loss: 0 transformer : 1
        elif 'multimodal' in self.training_strategy:
            training_type = int(self.training_strategy.split('_')[-1])
            add_decoder = training_type % 2
            add_transformer = training_type // 2
            if add_transformer:
                for name,param in self.model.named_parameters():
                    if 'transformer_blocks' in name:
                        print(f"add:{name}")
                        params.append(param)
                else:
                    for param in self.model.parameters():
                        param.requires_grad = False
            if add_decoder:
                params = params + list(self.global_condition.get_parameters(self.training_strategy))
                if not add_transformer:
                    self.model.eval()
        opt = torch.optim.AdamW(params,lr=lr)
        if hasattr(self.loss_fn,'discriminator'):
            opt_disc = torch.optim.AdamW(self.loss_fn.discriminator.parameters(),lr=2e-4,betas=(0.5,0.9))
        else:
            opt_disc = None
            disc_scheduler = None
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
            if hasattr(self.loss_fn,'discriminator') and hasattr(self.loss_fn.discriminator,'scheduler'):
                scheduler.append(
                    {
                        'scheduler':LambdaLR(opt_disc,lr_lambda=self.loss_fn.discriminator.scheduler.schedule),
                        'interval':'step',
                        'frequency':1
                    }
                )
            if not opt_disc is None:
                return [opt,opt_disc],scheduler
            return [opt],scheduler
        if not opt_disc is None:
            return [opt,opt_disc]
        return opt
        
    def p_mean_variance(self,x,c,t,clip_denoised:bool,return_codebook_ids=False,quantize_denoised=False,
                        return_x0=False,score_corrector=None,corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x,t_in,c,return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self,model_out,x,t,c,**corrector_kwargs)

        if return_codebook_ids:
            model_out,logits = model_out
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x,t=t,noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1.,1.)
        # if quantize_denoised:
        #     x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean,posterior_variance,posterior_log_variance = self.q_posterior(x_start=x_recon,x_t=x,t=t)
        if return_codebook_ids:
            return model_mean,posterior_variance,posterior_log_variance,logits
        elif return_x0:
            return model_mean,posterior_variance,posterior_log_variance,x_recon
        else:
            return model_mean,posterior_variance,posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self,x,c,t,clip_denoised=False,repeat_noise=False,
                 return_codebook_ids=False,quantize_denoised=False,return_x0=False,
                 temperature=1.,noise_dropout=0.,score_corrector=None,corrector_kwargs=None):
        b,*_,device = *x.shape,x.device
        outputs = self.p_mean_variance(x=x,c=c,t=t,clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean,_,model_log_variance,x0 = outputs
        else:
            model_mean,_,model_log_variance = outputs
        
        noise = noise_like(x.shape,device,repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise,p=noise_dropout)
        #no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
    @torch.no_grad()
    def progressive_denoising(self,cond,shape,verbose=True,callback=None,quantize_denoised=False,
                              img_callback=None,mask=None,x0=None,temperature=1.,noise_dropout=0.,
                              score_corrector=None,corrector_kwargs=None,batch_size=None,x_T=None,start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond,dict):
                cond = {key:cond[key][:batch_size] if not isinstance(cond[key],list) else
                        list(map(lambda x:x[:batch_size],cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond,list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
                    range(0,timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        
        for i in iterator:
            ts = torch.full((b,),i,device=self.device,dtype=torch.long)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,t=tc,noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0,ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback:img_callback(img,i)
        return img,intermediates
    
    def p_sample_loop(self,cond,shape,return_intermediates=False,
                      x_T=None,verbose=True,callback=None,timesteps=None,quantize_denoised=False,
                      mask=None,x0=None,img_callback=None,start_T=None,log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape,device=device)
        else:
            img = x_T
        
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        
        if start_T is not None:
            timesteps = min(timesteps,start_T)
        iterator = tqdm(reversed(range(0,timesteps)),desc='Sampling t',total=timesteps) if verbose else reversed(
            range(0,timesteps))
        
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        
        for i in iterator:
            ts = torch.full((b,),i,device=device,dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if return_intermediates:
            return img,intermediates
        return img
    
    @torch.no_grad()
    def sample(self,
                cond,
                x=None,
                range_image=None,
                uc=None,
                N=25,
                shape=None,
                batch_size=1,
                **kwargs):
        randn = torch.randn(N*batch_size,*shape).to(self.device)
        cond_mask = torch.zeros(N*batch_size).to(self.device)
        if self.replace_cond_frames:
            assert self.fixed_cond_frames
            cond_indices = self.fixed_cond_frames
            cond_mask = rearrange(cond_mask,"(b t) -> b t",t=self.movie_len)
            cond_mask[:,cond_indices] = 1
            cond_mask = rearrange(cond_mask,"b t -> (b t)")
        denoiser = lambda x,range_image,sigma,c,cond_mask: self.denoiser(self.model,x,range_image,sigma,c,N,self.action_former,cond_mask,**kwargs)
        samples = self.sampler(
            denoiser,randn,cond,uc=uc,cond_x=x,cond_range_image=range_image,cond_mask=cond_mask
        )
        return samples
    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.downsample_image_size)
            samples,intermediates = ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,intermediates
    
    @torch.no_grad()
    def sample_video(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.image_size)
            samples = ddim_sampler.sample_video(ddim_steps,batch_size,shape,cond,verbose=False,movie_len=self.movie_len,**kwargs)
            return samples
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample_video_log(self,cond,batch_size,ddim,ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels,) + tuple(self.image_size)
            samples,intermediates = ddim_sampler.sample(ddim_steps,batch_size*self.movie_len,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            samples,intermediates = self.sample(cond=cond,batch_size=batch_size,
                                                return_intermediates=True,**kwargs)
        return samples,intermediates
    
    @torch.no_grad()
    def log_images(self,batch,N=8,n_row=4,sample=True,ddim_steps=200,ddim_eta=1.,return_keys=None,
                   quantize_denoised=True,inpaint=True,plot_denoise_rows=False,plot_progressive_rows=True,
                   plot_diffusion_rows=True,**kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z,lidar_z,x,x_rec,range_image,lidar_rec,c = self.get_input(batch,return_first_stage_outputs=True)
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['inputs'] = x
        log['reconstruction'] = x_rec
        if not range_image is None:
            log['lidar_inputs'] = range_image
            log['lidar_reconstruction'] =  lidar_rec
        log['dense_range_image'] = c['dense_range_image']

        if plot_diffusion_rows:
            #get diffusion row
            diffusion_row = list()
            lidar_diffusion_row = list()
            z_start = z[:n_row]
            lidar_start = lidar_z[:n_row]
            
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]),'1 -> b',b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start,t=t,noise=noise)
                    lidar_noisy = self.q_sample(x_start=lidar_start,t=t,noise=noise)
                    diffusion_row.append(self.global_condition.decode_first_stage_interface('reference_image',z_noisy))
                    lidar_diffusion_row.append(self.global_condition.decode_first_stage_interface('lidar',lidar_noisy))


            diffusion_row = torch.stack(diffusion_row) # n_log_step,n_row,C,H,W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log['diffusion_row'] = diffusion_grid
            lidar_diffusion_row = torch.stack(lidar_diffusion_row)
            lidar_diffusion_grid = rearrange(lidar_diffusion_row,'n b c h w -> b n c h w') 
            lidar_diffusion_grid = rearrange(lidar_diffusion_grid,'b n c h w -> (b n) c h w')
            lidar_diffusion_grid = make_grid(lidar_diffusion_grid,nrow=lidar_diffusion_row.shape[0])
            log['lidar_diffusion_row'] = lidar_diffusion_grid

        if sample:
            # get denoise row
            c['hdmap'] = c['hdmap'][:N]
            c['boxes_emb'] = c['boxes_emb'][:N]
            c['text_emb'] = c['text_emb'][:N]
            c['dense_range_image'] = c['dense_range_image'][:N]
            c['x_start'] = z[:N]
            with self.ema_scope("Plotting"):
                samples,z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            samples,sample_lidar = torch.chunk(samples,2,dim=0)
            x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
            lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
            log["samples"] = x_samples
            log['lidar_samples'] = lidar_sample
            # if plot_denoise_rows:
            #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            #     log['denoise_row'] = denoise_grid
            
            # if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
            #         self.first_stage_model, IdentityFirstStage):
            #     # also display when quantizing x0 while sampling
            #     with self.ema_scope("Plotting Quantized Denoised"):
            #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
            #                                                  ddim_steps=ddim_steps,eta=ddim_eta,
            #                                                  quantize_denoised=True)
            #         # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
            #         #                                      quantize_denoised=True)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

            # if inpaint:
            #     #make a simple center square
            #     b,h,w = z.shape[0],z.shape[2],z.shape[3]
            #     mask = torch.ones(N,h,w).to(self.device)
            #     # zeros will be filled in
            #     mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            #     mask = mask[:, None, ...]
            #     with self.ema_scope("Plotting Inpaint"):

            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log['samples_inpainting'] = x_samples
            #     log["mask"] = mask

            #     #outpaint
            #     with self.ema_scope("Plotting Outpaint"):
            #         samples,_ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,eta=ddim_eta,
            #                                     ddim_steps=ddim_steps,x0=z[:N],mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_outpainting"] = x_samples
            # if plot_progressive_rows:
            #     with self.ema_scope("Plotting Progressives"):
            #         img, progressives = self.progressive_denoising(c,
            #                                                     shape=(self.channels, self.image_size[0],self.image_size[1]),
            #                                                     batch_size=N)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            # log["progressive_row"] = prog_row
            if return_keys:
                if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                    return log
                else:
                    return {key:log[key] for key in return_keys}
            return log
    @torch.no_grad()
    def log_video(self,batch,N=8,n_row=4,sample=True,**kwargs):
        log = dict()
        x = batch['reference_image']
        b = x.shape[0]
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        batch = {k:v[:N].to(self.global_condition.device) if isinstance(v,torch.Tensor) else v[:N] for k,v in batch.items()}
        z,lidar_z,x,x_rec,range_image,range_image_rec,c = self.get_input(batch,return_first_stage_outputs=True)

        log['inputs'] = x.reshape((N,-1)+x.shape[1:])
        log['reconstruction'] = x_rec.reshape((N,-1)+x_rec.shape[1:])
        if not range_image is None:
            log["lidar_inputs"] = range_image.reshape((N,-1)+range_image.shape[1:])
            log['lidar_reconstruction'] = range_image_rec.reshape((N,-1)+range_image_rec.shape[1:])
        # log['dense_range_image'] = c['dense_range_image'].reshape((N,-1),c['dense_range_image'].shape[1:])
        if sample:
            condition_key = c.keys()
            if 'hdmap' in condition_key:
                c['hdmap'] = c['hdmap'][:N*self.movie_len]
            if 'boxes_emb' in condition_key:
                c['boxes_emb'] = c['boxes_emb'][:N*self.movie_len]
            if 'text_emb' in condition_key:
                c['text_emb'] = c['text_emb'][:N*self.movie_len]
            if 'dense_range_image' in condition_key:
                c['dense_range_image'] = c['dense_range_image'][:N*self.movie_len]
            if 'actions' in condition_key:
                c['actions'] = c['actions'][:N]
            c['x_start'] = z[:N*self.movie_len]
            with self.ema_scope("Plotting"):
                # batch = {k:v.to(self.model.device) for k,v in batch.items()}
                samples = self.sample(c,z,lidar_z,shape=z.shape[1:],N=self.movie_len,batch_size=b)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            # print(torch.max(samples[0] - z[0]))
            samples = samples.to(self.global_condition.device)
            if not range_image is None:
                samples,sample_lidar = torch.chunk(samples,2,dim=0)
                x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
                lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
                log["samples"] = x_samples.reshape((N,-1)+x_samples.shape[1:])
                log['lidar_samples'] = lidar_sample.reshape((N,-1)+lidar_sample.shape[1:])
            else:
                x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
                log["samples"] = x_samples.reshape((N,-1)+x_samples.shape[1:])
        return log

    def get_infer_cond(self,batch):
        b = batch['reference_image'].shape[0]
        ref_img = super().get_input(batch,'reference_image') # (b c h w)
        hdmap = super().get_input(batch,'HDmap')# (b c h w)
        range_image = super().get_input(batch,'range_image') # (b c h w)
        dense_range_image = super().get_input(batch,'dense_range_image') # (b c h w)
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

        condition_out = self.global_condition.get_conditions(batch,return_encoder_posterior=True)
        z = condition_out['ref_image']
        lidar_z = condition_out['range_image']
        
        out = [z,lidar_z]
        c = {}
        c['hdmap'] = condition_out['hdmap'].to(torch.float32)
        c['boxes_emb'] = condition_out['boxes_emb'].to(torch.float32)
        c['text_emb'] = condition_out['text_emb'].to(torch.float32)
        c['dense_range_image'] = condition_out['dense_range_image']
        if self.predict:
            c['actions'] = condition_out['actions']
        out.extend([batch['reference_image'],batch['range_image']])
        out.append(c)
        return out

    def single_infer(self,batch,ddim_steps=200,ddim_eta=1.):
        use_ddim = ddim_steps is not None
        b = batch['reference_image'].shape[0]
        z,lidar_z,x_start,range_image_start,c = self.get_infer_cond(batch)
        log = {}
        log['inputs'] = x_start.reshape((b,-1)+x_start.shape[1:])
        log['lidar_inputs'] = range_image_start.reshape((b,-1)+range_image_start.shape[1:])
        c['x_start'] = z
        c['lidar_start'] = lidar_z
        samples = self.sample_video(cond=c,batch_size=b,ddim=use_ddim,ddim_steps=ddim_steps,eta=ddim_eta)
        samples,sample_lidar = torch.chunk(samples,2,dim=0)
        x_samples = self.global_condition.decode_first_stage_interface('reference_image',samples)
        lidar_sample = self.global_condition.decode_first_stage_interface('lidar',sample_lidar)
        log['samples'] = x_samples.reshape((b,-1)+x_samples.shape[1:])
        log['lidar_samples'] = lidar_sample.reshape((b,-1)+lidar_sample.shape[1:])
        return log

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/svd_range_image.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    network = instantiate_from_config(cfg['model'])#.cuda()#.to('cuda:7')
    
    x = torch.randn((2,5,128,256,3))#.cuda()
    # x.requires_grad_(True)
    hdmap = torch.randn((2,5,128,256,3))#.cuda()
    boxes = torch.randn((2,5,70,16))#.cuda()
    text = [["None" for k in range(5)] for j in range(2)]
    box_category = [[["None" for k in range(70)] for i in range(5)] for j in range(2)]
    # depth_cam_front_img = torch.randn((2,5,128,256,3))
    import matplotlib.image as mpimg
    from PIL import Image
    # range_image = mpimg.imread('000.jpg')
    # range_image = Image.fromarray(range_image[:,:,:3])
    # range_image = np.array(range_image.resize((256,128)))
    # range_image = torch.tensor(range_image,dtype=torch.float32)
    # range_image = range_image.unsqueeze(0)
    # range_image = range_image.unsqueeze(0)
    range_image = torch.randn((2,5,128,256,3))#.cuda()
    dense_range_image = torch.randn((2,5,128,256,3))#.cuda()
    # depth_cam_front_img = torch.randn((2,5,128,256,3))
    actions = torch.randn((2,5,14))#.cuda()
    # out = {'text':text,
    #        '3Dbox':boxes,
    #        'category':box_category,
    #        'reference_image':x,
    #        'HDmap':hdmap,
    #        "range_image":range_image,
    #        'dense_range_image': dense_range_image,
    #        'actions':actions}
    bev_images = torch.randn((2,5,128,256,3))#.cuda()
    out = {
        # 'text':text,
           'reference_image':x,
           "range_image":range_image,
           'actions':actions,
           'HDmap':hdmap,
           '3Dbox':boxes,
           'category':box_category,
           'dense_range_image':dense_range_image
        }
    
    # network.log_video(out)
    # network.configure_optimizers()
    # loss = network.log_video(out)
    # loss.backward()
    # print(network.model.get_last_layer().shape)
    # loss,loss_dict = network.shared_step(out,optimizer_idx=0)
    # print(loss)
    # loss.backward()
    loss,loss_dict = network.shared_step(out)
    # network.configure_optimizers()
    # network.shared_step(out)
    # loss,loss_dict = network.validation_step(out,0)
    # loss.backward()
    # opt = network.configure_optimizers()
    # opt.step()
    pass


