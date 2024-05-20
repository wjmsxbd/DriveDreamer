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
from ldm.util import log_txt_as_img,exists,default,ismap,isimage,mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGuassianDistribution
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.attention import PositionalEncoder
from ldm.modules.diffusionmodules.util import extract_into_tensor
import omegaconf

__conditioning_keys__ = {'concat':'c_concat',
                         'crossattn':'c_crossattn',
                         'adm':'y'}

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1,r2,shape,device):
    return (r1-r2) * torch.rand(*shape,device=device) + r2


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
                 use_ema=True,
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
        self.model = DiffusionWrapper(unet_config)

        count_params(self.model,verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
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
                t = repeat(torch.tensor([t],'1 -> b',b = n_row))
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
    
class DiffusionWrapper(pl.LightningModule):
    def __init__(self,diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
    
    def from_pretrained_model(self,model_path):
        self.diffusion_model.from_pretrained_model(model_path)

    def forward(self,x,t,boxes_emb,text_emb):
        return self.diffusion_model(x,t,boxes_emb=boxes_emb,text_emb=text_emb)

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
        n_imgs_per_orw = len(denoise_row)
        denoise_row = torch.stack(denoise_row) #n_log_step n_row C H W
        denoise_grid = rearrange(denoise_row,'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid,'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid,nrow=n_imgs_per_orw)
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
    def validation_step(self,batch,batch_idx):
        _,loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _,loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key+'_ema':loss_dict_ema[key] for key in loss_dict_ema}
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
        x = x / 255.
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
        x = ref_img
        z = self.ref_img_encoder.encode(x)
        z = self.get_first_stage_encoding(z)
        out = [z]
        c = {}
        c['hdmap'] = hdmap
        c['boxes'] = boxes
        c['category'] = boxes_category
        c['text'] = text
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
        x = ref_img
        x = self.get_first_stage_encoding(self.ref_img_encoder.encode(x))
        c = {}
        c['hdmap'] = hdmap
        boxes = rearrange(batch['3Dbox'],'b n c d -> (b n) c d')
        boxes_category = rearrange(batch['category'],'b n c d -> (b n) c d')
        c['boxes'] = boxes
        c['category'] = boxes_category
        c['text'] = rearrange(batch['text'],'b n d -> (b n) d').unsqueeze(1)
        loss,loss_dict = self(x,c)
        return loss,loss_dict

    def forward(self,x,c,*args,**kwargs):
        t = torch.randint(0,self.num_timesteps,(x.shape[0],)).long()#device=self.device
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
                z = self.get_first_stage_encoding(self.ref_img_encoder.encode(z))
                hdmap = hdmap_list[i]
                hdmap = self.get_first_stage_encoding(self.hdmap_encoder.encode(hdmap))
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
            hdmap = self.get_first_stage_encoding(self.hdmap_encoder.encode(cond['hdmap']))
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
    
    def decode_first_stage(self,z):
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
            return self.ref_img_encoder.decode(z)
    
    def differentiable_decode_first_stage(self,z):
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
            for param in self.model.parameters():
                if 'temporal' in param or 'gated' in param:
                    print(f"model:add {param} into optimizers")
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
                 tempperature=1.,noise_dropout=0.,score_corrector=None,corrector_kwargs=None):
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
            model_mean,_,mean_log_variance,x0 = outputs
        else:
            model_mean,_,model_log_variance = outputs
        
        noise = noise_like(x.shape,device,repeat_noise) * tempperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise,p=noise_dropout)
        #no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b,*((1,) * len(x.shape) - 1))
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

            img,x0_partial = self.p_sample(img,cond,ts,clip_denoised=self.clip_denoised,
                                           quantize_denoised=quantize_denoised,
                                           temperature=temperature[i],noise_dropout=noise_dropout)
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
            shape = (batch_size,self.channels) + self.image_size
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
                mask = torch.onse(N,h,w).to(self.device)
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
                                                                shape=(self.channels, self.image_size, self.image_size),
                                                                batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row
            if return_keys:
                if np.intersect1d(list(log.keys()),return_keys).shape[0] == 0:
                    return log
                else:
                    return {key:log[key] for key in return_keys}
            return log
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/first_stage_step1_config_online2.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    network = instantiate_from_config(cfg['model'])
    x = torch.randn((2,2,448,768,3))
    hdmap = torch.randn((2,2,448,768,4))
    text = torch.randn((2,2,768))
    boxes = torch.randn((2,2,50,16))
    box_category = torch.randn(2,2,50,768)
    out = {'text':text,
           '3Dbox':boxes,
           'category':box_category,
           'reference_image':x,
           'HDmap':hdmap}
    network.log_images(out)


