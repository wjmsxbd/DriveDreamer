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
import re
import copy
from PIL import Image

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
                 scheduler_config=None,
                 copy_ca_weight=False,
                 slow_fast_path=None,
                 num_cameras=1):
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
        self.copy_ca_weight = copy_ca_weight
        self.scheduler_config = scheduler_config
        self.num_cameras = num_cameras
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
        if not slow_fast_path is None:
            self.init_from_ckpt(slow_fast_path,ignore_keys,load_from_ema,True)

    def init_first_stage(self,config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def init_from_ckpt(self,path=None,ignore_keys=[],load_from_ema=False,load_from_slow_fast=False):
        sd = torch.load(path,map_location='cpu')
        if 'state_dict' in list(sd.keys()):
            sd = sd['state_dict']
        if load_from_slow_fast:
            sd_replace = {}
            for key,value in sd.items():
                sd_replace[key[6:]] = value
            sd = sd_replace
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
            sd['model.diffusion_model.input_blocks.0.0.weight'] = copy.deepcopy(param)
            if self.copy_ca_weight:
                find_keys = ["attn2","norm2"]
                replace_keys = ["attn3","norm4"]
                for i in range(len(find_keys)):
                    pattern = r"model\.diffusion_model\.[^.]+?\.\d+(\.\d+)?\.transformer_blocks\.\d+\.{}\.[^\.]+".format(find_keys[i])
                    matched_strings = [s for s in list(sd.keys()) if re.match(pattern,s)]
                    print(matched_strings)
                    for key in matched_strings:
                        print("now process"+key)
                        value = sd[key]
                        key_split = key.split('.')
                        if key_split[2] == 'middle_block':
                            key_split[6] = replace_keys[i]
                        else:
                            key_split[7] = replace_keys[i]
                        new_key = '.'.join(key_split)
                        print("copy key"+new_key)
                        sd[new_key] = copy.deepcopy(value)

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
        if self.use_ema:
            with self.ema_scope():
                _,loss_dict_ema = self.shared_step(batch)
                loss_dict_ema = {key+"ema":loss_dict_ema[key] for key in loss_dict_ema}    
                self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        else:        
            _,loss_dict_no_ema = self.shared_step(batch)
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        

    def shared_step(self,batch):
        x = self.get_input(batch)
        if 'first_frame' in batch.keys():
            self.prepare_model_setting(batch['first_frame'])
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

    def clear_model_cache(self):
        self.model.clear_model_cache()

    def set_model_init_feature(self,flag):
        self.model.set_model_init_feature(flag)

    def pack_camera_squence(self,batch):
        for key in batch.keys():
            if isinstance(batch[key],torch.Tensor):
                batch[key] = rearrange(batch[key],'b n ... -> (b n) ...')
        return batch

    # input_shape: cond_frame tensor:(b h w c)
    #              image tensor:(b h w c)
    #              HDmap tensor:(b h w c)
    #              3Dbox List[str]:(b n) 
    # 多帧会出问题
    @torch.no_grad()
    def get_input(self,batch,return_first_stage_outputs=False,bs=None):
        if self.num_cameras == 6:
            self.pack_camera_squence(batch)
        x = batch[self.input_keys]
        if bs is not None:
            x = x[:bs]
        assert isinstance(x,torch.Tensor)
        encoder_posterior = self.encode_first_stage(x)
    
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if return_first_stage_outputs:
            x_rec = self.decode_first_stage(z)
            return z,x_rec
        return z
    
    
    def get_losses(self,x,cond,return_predict=False):
        return self.loss_fn._forward(self.model,self.denoiser,cond,x,return_predict)
    
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
        elif self.training_strategy == 'feature':
            for name,param in self.named_parameters():
                if name.startswith('model.diffusion_model.input_blocks'):
                    pass
                elif name.startswith("model.diffusion_model"):
                    # print(f"add:{name}")
                    params.append(param)

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
    def clear_model_cache(self):
        # call self.model.clear_model_cache()
        self.model.clear_model_cache()

    @torch.no_grad()
    def get_unconditional_conditioning(self,batch,is_inference=True):
        ucg_keys = [e.input_key for e in self.global_condition.embedders if e.ucg_rate>0.]
        c,uc = self.global_condition.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.global_condition.embedders)>0 else list(),
            is_inference=is_inference,
        )
        return c,uc
    
    def get_condition(self,batch):
        return self.global_condition(batch)

    def get_feature_cache(self):
        return self.model.get_feature_cache()
    
    def replace_feature_cache(self,feature):
        self.model.replace_feature_cache(feature)

    def infer(self,batch):
        ucg_keys = [e.input_key for e in self.global_condition.embedders if e.ucg_rate>0.]

        x = self.get_input(batch)
        c,uc = self.global_condition.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.global_condition.embedders)>0 else list()
        )
        with self.ema_scope("Plotting"):
            samples = self.sample(x.shape[0],c,uc,x.shape[1:])
        return samples

    def prepare_sigmas(self,):
        return self.sampler.prepare_sigmas()

    def infer_step(self,x,sigmas,sigma_step,cond,uc=None):
        uc = default(uc,cond)
        s_in = x.new_ones([x.shape[0]])
        num_sigmas = len(sigmas)
        gamma = self.sampler.get_gamma(num_sigmas,sigmas[sigma_step])
        if sigma_step == 0:
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2)
        denoiser = lambda input,sigma,c: self.denoiser(self.model,input,sigma,c)
        x = self.sampler.sampler_step(
            s_in * sigmas[sigma_step],
            s_in * sigmas[sigma_step+1],
            denoiser,
            x,
            cond,
            uc,
            gamma
        )
        return x

    def prepare_model_setting(self,first_frame):
        self.model.prepare_model_setting(first_frame)

    def get_zero_feature(self,):
        return self.model.get_zero_feature()

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
        N=6,
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
        input = batch['image']
        if len(input.shape) == 5:
            input = rearrange(input, "b n c h w -> (b n) c h w")
        log['inputs'] = input
        log['cond_frame'] = batch['cond_frames']
        x,x_rec = self.get_input(batch,return_first_stage_outputs=True)
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        # N = N
        # n_row = n_row
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
    
    @torch.no_grad()
    def log_latents(
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
        # log['cond_frame'] = batch['cond_frames']
        x,x_rec = self.get_input(batch,return_first_stage_outputs=True)
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        log['reconstruction'] = x_rec
        c,uc = self.global_condition.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.global_condition.embedders)>0 else list()
        )
        c['concat'] = torch.cat((batch['cond_frames'],c['concat']), dim=1)
        # print(c['concat'].shape)
        # print(batch['cond_frames'].shape)
        x = x[:N].to(self.device)
        for k in c:
            if isinstance(c[k],torch.Tensor):
                c[k],uc[k] = map(lambda y:y[k][:N].to(self.device),(c,uc))
        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    N,c,uc,x.shape[1:]
                )
                log['latent'] = samples
                samples = self.decode_first_stage(samples)
                log['samples'] = samples
        return log
    
from ldm.models.diffusion.slow_fast_learning import FeatureCache2D
class StreamingSDInferPipeLine(pl.LightningModule):
    def __init__(self,model_config,num_steps,use_feature_cache=False):
        super().__init__()
        self.model = instantiate_from_config(model_config)
        self.use_feature_cache= use_feature_cache
        if use_feature_cache:
            self.feature_cache = FeatureCache2D(num_steps)
        self.cond_frames = None
        self.noise_cache = None
        
    def save_tensor_as_image(self,tensor,file_path,index,frame):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.clamp(-1.,1.)
        tensor = (tensor + 1.) / 2.
        tensor = tensor * 255.0
        tensor = tensor.byte()

        for i in range(tensor.shape[0]):
            img = tensor[i]
            img = img.permute(1,2,0)
            img = img.numpy()
            img = Image.fromarray(img)
            save_file_path = os.path.join(file_path,f'{index:02d}_{frame+i:02d}.png')
            img.save(save_file_path)
    
    def save_tensor_as_MVimage(self,tensor,file_path,index,frame):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        h,w = tensor.shape[-2:]
        tensor = tensor.clamp(-1, 1)  # 确保值在[-1, 1]之间
        tensor = (tensor + 1.0) / 2.0  # 转换到[0, 1]
        tensor = tensor * 255.0  # 转换到[0, 255]
        tensor = tensor.byte()  # 转换为byte类型

        # 构建保存文件的路
        # 创建一个空白图片，用于存放拼接后的大图    
        big_image = Image.new('RGB', (3 * w, 2 * h), (255, 255, 255))  # 白色背景

        # 将张量转换为PIL图像，并拼接
        for j in range(tensor.shape[0]):
            save_file_path = os.path.join(file_path, f'{index:02d}_{frame+j:02d}.png')
            for i in range(tensor.shape[1]):  # 遍历6张图片
                img = tensor[j][i]  # 获取单张图片的张量
                img = img.permute(1, 2, 0)  # 调整维度为高度x宽度x通道
                img = img.numpy()  # 转换为numpy数组
                img = Image.fromarray(img)  # 转换为PIL图像

            # 计算图片在大图中的位置
                row = i // 3  # 行号
                col = i % 3  # 列号
                if row < 2:  # 只有两行
                    position = (col * w, row * h)  # 确定位置
                    big_image.paste(img, position)  # 粘贴图片

            # 保存大图片
            big_image.save(save_file_path)
        

    def decode_first_stage(self,latents,file_path,index,n_samples=8,decoder=None):
        latents = torch.stack(latents,dim=0)
        print(latents.shape)
        n_cam =1
        if len(latents.shape) == 5:
            n_cam = 6
        chunk_size = (latents.shape[0] + n_samples - 1) // n_samples
        latents_chunk = torch.chunk(latents,chunks=chunk_size,dim=0)
        start_frame = 0
        for chunk in latents_chunk:
            chunk = chunk.to(self.model.device)
            if n_cam > 1 :
                chunk = rearrange(chunk, "b n c h w -> (b n) c h w") 
            if not decoder is None:
                output = decoder.decode_first_stage(chunk)
            else:
                output = self.model.decode_first_stage(chunk)
            if n_cam > 1 :
                chunk = rearrange(chunk, "(b n) c h w -> b n c h w",n = n_cam) 
                output = rearrange(output, "(b n) c h w -> b n c h w",n = n_cam) 
                output = output.cpu()
                self.save_tensor_as_MVimage(output,file_path,index,frame=start_frame)
            else:
                output = output.cpu()
                self.save_tensor_as_image(output,file_path,index,frame=start_frame)
            chunk = chunk.cpu()
            start_frame += chunk.shape[0]
        


    def reset(self,init_feature):
        self.model.clear_model_cache()
        self.model.set_model_init_feature(init_feature)

    def set_init_feature(self,init_feature):
        self.model.set_model_init_feature(init_feature)

    def continue_infer(self,collate_latent,T,batch,first_frame_idx):
        last_frame = torch.stack([collate_latent[idx][-1] for idx in first_frame_idx],dim=0)
        cond_frame = last_frame.to(self.device)
        now_frame = len(collate_latent[first_frame_idx[0]])
        for i in range(T):
            sigmas = self.model.prepare_sigmas()
            num_sigmas = len(sigmas)
            c,uc = self.model.get_unconditional_conditioning(batch)
            z = torch.randn_like(cond_frame).to(self.device)
            uc['concat'][:,:4] = cond_frame
            c = uc
            for i in range(num_sigmas-1):
                if self.use_feature_cache:
                    feature_cache = self.feature_cache.get_feature_in_row(i)
                    self.model.replace_feature_cache(feature_cache)
                    z = self.model.infer_step(z,sigmas,i,c,uc)
                    feature_cache = self.model.get_feature_cache()
                    self.feature_cache.update(feature_cache,i)
                else:
                    z = self.model.infer_step(z,sigmas,i,c,uc)
            cond_frame = z
            batch_index = 0
            for idx in first_frame_idx:
                if now_frame == len(collate_latent[idx]):
                    collate_latent[idx].append(z[batch_index].detach().cpu())
                batch_index += 1
            now_frame += 1
        return collate_latent


    def _forward(self,batch,replace_cond_frames=False,return_first_frame=False):
        sigmas = self.model.prepare_sigmas()
        num_sigmas = len(sigmas)
        z = self.model.get_input(batch)
        c,uc = self.model.get_unconditional_conditioning(batch)
        if return_first_frame:
            z_ = z
        randn = torch.randn_like(z).to(z.device)
        z = randn
        if replace_cond_frames:
            c['concat'][:,:4] = self.cond_frames
        for i in range(num_sigmas-1):
            if self.use_feature_cache:
                feature_cache = self.feature_cache.get_feature_in_row(i)
                self.model.replace_feature_cache(feature_cache)
                self.model.prepare_model_setting(batch['first_frame'])
                z = self.model.infer_step(z,sigmas,i,c,uc)
                feature_cache = self.model.get_feature_cache()
                # self.feature_cache.update(feature_cache,i)
            else:
                z = self.model.infer_step(z,sigmas,i,c,uc)
        if return_first_frame:
            return z,z_
        else:
            return z

    def init_cache(self,batch):
        if self.feature_cache.get_feature_in_row(0) == []:
            copy_batch = {k:copy.deepcopy(v) for k,v in batch.items()}
            self.model.shared_step(copy_batch)
            self.feature_cache.init_cache(self.model)
            
    def forward(self,batch):
        if self.use_feature_cache:
            self.init_cache(batch)
        if batch['first_frame'][0] == [1] or batch['first_frame'] == 1:
            self.noise_cache = None
            output,z = self._forward(batch,False,True)
            self.cond_frames = z.detach()
        else:
            output = self._forward(batch,True)
            self.cond_frames = output.detach()
        return output.detach().cpu()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/StreamingSD_cache.yaml',
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
    first_frame = [1,1]
    out = {
        'first_frame':first_frame,
        'image':x,
        'cond_frames':cond_frames,
        'HDmap':hdmap,
        '3Dbox':boxes
    }
    network.configure_optimizers()
    # loss,loss_dict = network.shared_step(out)
    # log = network.log_images(out)
