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
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import AutoencoderKL,VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.attention import PositionalEncoder
from ldm.modules.diffusionmodules.util import extract_into_tensor
import omegaconf
import copy
from typing import Iterable,List,Union,Optional,Dict,Tuple


def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1,r2,shape,device):
    return (r1-r2) * torch.rand(*shape,device=device) + r2

__conditioning_keys__ = {'concat':'c_concat',
                         'crossattn':'c_crossattn',
                         'adm':'y'}


class StreamingSD(pl.LightningModule):
    def __init__(self,
                 global_condition_config,
                 first_stage_config,
                 unet_config,
                 input_keys='image',
                 concat_mode=True,
                 scale_factor=1.0,
                 scale_by_std=False,
                 sampler_config=None,
                 denoiser_config=None,
                 loss_fn_config=None,
                 training_strategy="full",
                 load_from_ema=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor="val/loss",
                 use_ema=False,
                 use_scheduler=True,
                 scheduler_config=None):
        super().__init__()
        self.global_condition = instantiate_from_config(global_condition_config)
        self.model = instantiate_from_config(unet_config)
        self.init_first_stage(first_stage_config)
        self.scale_by_std = scale_by_std
        self.use_scheduler = use_scheduler
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor",torch.tensor(scale_factor))
        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.loss_fn = instantiate_from_config(loss_fn_config)
        self.use_ema = use_ema
        self.scheduler_config = scheduler_config
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")
        self.monitor = monitor
        self.training_strategy = training_strategy
        self.concat_mode = concat_mode
        self.input_keys = input_keys
        self.restart_from_ckpt = False
        if not ckpt_path is None:
            self.init_from_ckpt(ckpt_path,ignore_keys,load_from_ema)
            self.restart_from_ckpt = True

    def init_first_stage(self,config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def init_from_ckpt(self,path=None,ignore_keys=[],load_from_ema=False):
        sd = torch.load(path,map_location='cpu')
        if 'state_dict' in list(sd.keys()):
            sd = sd['state_dict']
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
        if self.concat_mode and path == 'stable_diffusion/sd-v1-4.ckpt':
            #TODO: modify input_block.conv value
            param = sd['model.diffusion_model.input_blocks.0.0.weight']
            param_pad = torch.zeros((param.shape[0],8)+param.shape[2:])
            param = torch.cat([param,param_pad],dim=1)
            sd['model.diffusion_model.input_blocks.0.0.weight'] = param

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict".format(k))
                    del sd[k]
        missing,unexpected = self.load_state_dict(sd,strict=False)
        print(f"Restore from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def training_step(self,batch,batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.log("global_step",self.global_step,prog_bar=True,logger=True,on_step=True,on_epoch=False)

        if self.use_scheduler:
            opt = self.optimizers()
            if isinstance(opt,list):
                lr = opt[0].param_groups[0]['lr']
            else:
                lr = opt.param_groups[0]['lr']
            self.log("lr_abs",lr,prog_bar=True,logger=True,on_step=True,on_epoch=False,)
        return loss
    
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        _,loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _,loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key+"ema":loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def shared_step(self,batch):
        x = self.get_input(batch)
        loss,loss_dict = self(x,batch)
        return loss,loss_dict

    @contextmanager
    def ema_scope(self,context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def encode_first_stage(self,x):
        return self.first_stage_model.encode(x) 
    
    def decode_first_stage(self,z):
        z  = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def get_first_stage_encoding(self,encoder_posterior):
        if isinstance(encoder_posterior,DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior,torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    # input_shape: cond_frame tensor:(b h w c)
    #              image tensor:(b h w c)
    #              HDmap tensor:(b h w c)
    #              3Dbox List[str]:(b n) 
    # 多帧会出问题
    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False,bs=None):
        x = batch[self.input_keys]
        if bs is not None:
            x = x[:bs]
        assert isinstance(x,torch.Tensor)
        encoder_posterior = self.encode_first_stage(x)
        #FX TODO:call self.model.clear_model_cache() if batch['first_frame'] == 1

        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if return_first_stage_outputs:
            x_rec = self.decode_first_stage(z)
            return z,x_rec
        return z
        
    def on_train_batch_end(self,*args,**kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def forward(self,x,batch):
        loss = self.loss_fn(self.model,self.denoiser,self.global_condition,x,batch)
        log_prefix = "train" if self.training else "val"
        loss_dict = {f"{log_prefix}/loss":loss}
        return loss,loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list()
        if self.training_strategy == 'full':
            params = params + list(self.model.parameters())
        else:
            raise NotImplementedError
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
                }
            ]
            return [opt],scheduler
        return opt

    #FX TODO: clear feature cache
    def clear_model_cache(self,):
        # call self.model.clear_model_cache()
        pass

    @torch.no_grad()
    def sample(
        self,
        bs,
        cond:Dict,
        uc:Union[Dict,None]=None,
        shape:Union[Tuple,List,None]=None,
        **kwargs,
        ):
        randn = torch.randn(bs,*shape).to(self.device)
        denoiser = lambda input,sigma,c: self.denoiser(self.model,input,sigma,c,**kwargs)
        samples = self.sampler(denoiser,randn,cond,uc=uc,)
        return samples

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample:bool=True,
        ucg_keys:List[str]=None,
        **kwargs):
        #TODO: add unconditional_sampler
        conditioner_input_keys = [e.input_key for e in self.global_condition.embedders if e.ucg_rate>0.]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys, "
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()
        log['inputs'] = batch['image']
        log['cond_frame'] = batch['cond_frames']
        x,x_rec = self.get_input(batch,return_first_stage_outputs=True)
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['reconstruction'] = x_rec
        c,uc = self.global_condition.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.global_condition.embedders)>0 else list()
        )
        x = x[:N].to(self.device)
        for k in c:
            if isinstance(c[k],torch.Tensor):
                c[k],uc[k] = map(lambda y:y[k][:N].to(self.device),(c,uc))
        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    N,c,uc,x.shape[1:]
                )
                samples = self.decode_first_stage(samples)
                log['samples'] = samples
        return log
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/StreamingSD.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    network = instantiate_from_config(cfg['model'])#.cuda()#.to('cuda:7')
    x = torch.randn((2,3,128,256))#.cuda()
    # x.requires_grad_(True)
    hdmap = torch.randn((2,3,128,256))#.cuda()
    boxes = [["None" for k in range(30)] for i in range(2)]
    cond_frames = torch.randn((2,3,128,256))
    out = {
        'image':x,
        'cond_frames':cond_frames,
        'HDmap':hdmap,
        '3Dbox':boxes
    }
    # loss,loss_dict = network.shared_step(out)
    log = network.log_images(out)