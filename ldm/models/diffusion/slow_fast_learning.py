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
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class SlowLearning(nn.Module):
    def __init__(self,
                ):
        super().__init__()
        self.cache = FeatureCache2D(num_steps=1)
        
    def forward(self,model,cond):
        x = cond['image']
        feature_cache = self.cache.get_feature_in_row(0)
        if feature_cache != []:
            model.replace_feature_cache(feature_cache)
        model.prepare_model_setting(cond['first_frame'])
        loss = model.get_losses(x,cond)
        feature_cache = model.get_feature_cache()
        self.cache.update(feature_cache,0)
        return loss

class FeatureCache1D:
    def __init__(self,window_size=10,choose_feature_idx=[-10,-5,1]):
        self.window_size = window_size
        self.choose_feature_idx = choose_feature_idx
        self.cache = []
        self.first = []

    def replace_cache(self,feature,index=None):
        if index is None:
            self.cache = copy.deepcopy(feature)
        else:
            for i in range(self.window_size): # time 
                for j in range(len(self.cache[0])): # layer
                    self.cache[i][j][index] = copy.deepcopy(feature[i][j])

    def check_cache_is_empty(self,):
        return self.cache == []

    def clear_cache(self):
        self.cache = []

    def get_cache(self):
        return copy.deepcopy(self.cache)
    
    def update(self,feature):
        assert len(self.cache) == self.window_size
        self.cache.pop(0)
        self.cache.append(copy.deepcopy(feature))

    def get_feature(self,col):
        temp_feature = []
        for idx in self.choose_feature_idx:
            temp_feature.append(self.cache[idx][col])
        temp_feature.append(self.first[col])
        temp_feature = torch.stack(temp_feature,dim=1)
        return copy.deepcopy(temp_feature)

    

class FeatureCache2D:
    def __init__(self,num_steps):
        self.cache = [[] for i in range(num_steps)]
        self.num_steps = num_steps

    def get_feature_in_row(self,row):
        return copy.deepcopy(self.cache[row])

    def update(self,feature,row):
        self.cache[row] = copy.deepcopy(feature)

    def clear_feature_cache(self,):
        self.cache.clear()



class FastLearning(nn.Module):
    def __init__(self,num_steps):
        super().__init__()
        self.cache = FeatureCache2D(num_steps)
    
    @torch.no_grad()
    def forward(self,model,batch,replace):
        sigmas = model.prepare_sigmas()
        num_sigmas = len(sigmas)
        c,uc = model.get_unconditional_conditioning(batch)
        z = model.get_input(batch)
        c['image'] = z
        randn = torch.randn_like(z).to(z.device)
        z = randn
        for i in range(len(replace)):
            if replace[i]:
                c['concat'][i,:4] = batch['samples'][i]
        for i in range(num_sigmas-1):
            feature_cache = self.cache.get_feature_in_row(i)
            if feature_cache != []:
                model.replace_feature_cache(feature_cache)
            model.prepare_model_setting(batch['first_frame'])
            z = model.infer_step(z,sigmas,i,c,uc)
            feature_cache = model.get_feature_cache()
            self.cache.update(feature_cache,i)
        
        return z,c

class FrameCounter:
    def __init__(self):
        self.num_frame = None

    def check_is_empty(self,):
        return self.num_frame is None

    def get_num_frames(self):
        return self.num_frame

    def update(self):
        for i in range(len(self.num_frame)):
            self.num_frame[i] += 1

    def prepare(self,first_frame):
        if self.num_frame is None:
            self.num_frame = []
            for i in range(len(first_frame)):
                self.num_frame.append(0)
        else:
            for i in range(len(first_frame)):
                if first_frame[i] == [1]:
                    self.num_frame[i] = 0

    def replace(self,window_size):
        replace_flags = []
        for i in range(len(self.num_frame)):
            if self.num_frame[i] % window_size != 0:
                replace_flags.append(True)
            else:
                replace_flags.append(False)
        return replace_flags

    

class SlowFastLearning(pl.LightningModule):
    def __init__(self,model_config,num_steps,window_size=1,monitor='val/loss',use_scheduler=False,scheduler_config=None,force_train_step=None,replace_window_size=2):
        super().__init__()
        self.automatic_optimization = False
        self.model = instantiate_from_config(model_config)
        self.generate_data = []
        self.use_scheduler = use_scheduler
        self.scheduler_config = scheduler_config
        self.num_steps = num_steps
        self.window_size = window_size
        self.replace_window_size = replace_window_size
        self.slow_learning = SlowLearning()
        self.fast_learning = FastLearning(num_steps)
        self.monitor = monitor
        #TODO: adaptive window size add monitor
        self.adaptive_window_size = [1,3,5,8,21,34,55,1000]
        self.adaptive_point = 0
        self.num_frame = FrameCounter()
        self.cond_frames = None
        self.force_train_step = force_train_step
        
    def replace_cond_latent(self,cond,output,first_frame):
        for i in range(len(first_frame)):
            if first_frame[i] == [1]:
                continue
            cond['concat'][i,:4] = output[i].detach()
        return cond

    def clear_generate_data(self,):
        self.generate_data.clear()

    def on_train_batch_start(self,batch,batch_idx,current_epoch):
        if self.adaptive_window_size[self.adaptive_point] == self.current_epoch:
            self.adaptive_point += 1
            self.replace_window_size *= 2

    def gather_dataset_len(self,dataset_len):
        rank = int(os.environ.get("RANK", 0)) 
        world_size = int(os.environ.get("WORLD_SIZE", 1)) 
        gathered_lengths = [torch.zeros(1,dtype=torch.long,device=self.device) for _ in range(world_size)]
        dist.all_gather(gathered_lengths,torch.tensor([dataset_len],dtype=torch.long,device=self.device))
        return gathered_lengths
    
    def collate_fn(self,batch):
        out = {}
        batch = batch[0]
        if isinstance(batch,dict):
            return batch
        else:
            return None

    #TODO: choose training type and manual_backward optimizer.step() optimizer.zero_grad()
    def training_step(self,batch,batch_idx):
        self.force_training_step(batch,batch_idx)

    def prepare_model_setting(self,first_frame):
        self.model.prepare_model_setting(first_frame)

    def force_training_step(self,batch,batch_idx):
        assert 'first_frame' in batch.keys()
        self.num_frame.prepare(batch['first_frame'])
        if len(self.generate_data) == self.force_train_step:
            # slow learning
            b = len(batch['first_frame'])
            fast_dataset = self.generate_data
            fast_dataset = DataLoader(fast_dataset,batch_size=1,collate_fn=self.collate_fn)
            self.model.train()
            tqdm_bar = tqdm(enumerate(fast_dataset),total=len(fast_dataset))
            for _,data in tqdm_bar:
                for k in data.keys():
                    if isinstance(data[k],torch.Tensor):
                        data[k] = data[k].to(self.device)
                loss = self.slow_learning(self.model,data)
                log_prefix = "train" if self.training else 'val'
                loss_dict = {f"{log_prefix}/loss":loss}
                tqdm_bar.set_postfix(loss=loss.item())
                self.log_dict(loss_dict,prog_bar=True,logger=True,on_step=True,on_epoch=True)
                self.slow_optimizer.zero_grad()
                self.manual_backward(loss)
                self.slow_optimizer.step()
            self.clear_generate_data()
            # fast learning
            self.model.eval()
            batch['samples'] = self.cond_frames
            output,cond = self.fast_learning(self.model,batch,self.num_frame.replace(self.replace_window_size))
            cond['first_frame'] = batch['first_frame']
            cond = self.replace_cond_latent(cond,self.cond_frames,batch['first_frame'])
            self.cond_frames = output.detach()
            cond = {k:copy.deepcopy(v.detach().cpu()) if isinstance(v,torch.Tensor) else v for k,v in cond.items()}
            self.generate_data.append(cond)
        else:
            #fast learning
            batch['samples'] = self.cond_frames
            output,cond = self.fast_learning(self.model,batch,self.num_frame.replace(self.replace_window_size))
            cond['first_frame'] = batch['first_frame']
            cond = self.replace_cond_latent(cond,self.cond_frames,batch['first_frame'])
            self.cond_frames = output.detach()
            cond = {k:copy.deepcopy(v.detach().cpu()) if isinstance(v,torch.Tensor) else v for k,v in cond.items()}
            self.generate_data.append(cond)
        self.num_frame.update()


        
    def manual_backward(self,loss,*args,**kwargs):
        if not loss is None:
            loss.backward()
        for param_group in self.slow_optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param,device=self.device)
                assert isinstance(param.grad,torch.Tensor),f"the type is {type(param.grad)}"
                dist.all_reduce(param.grad,op=dist.ReduceOp.SUM)

        # for param in self.parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for param {param.name}: {param.grad}")
        #         break

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample:bool=True,
        ucg_keys:List[str]=None,
        **kwargs
    ):
        return self.model.log_images(batch,N,n_row,sample,ucg_keys,**kwargs)

    def get_adaptive_weight(self,):
        return torch.exp(-torch.tensor(self.num_frame) / 256)

    def validation_step(self,batch,batch_idx):
        assert 'first_frame' in batch.keys()
        if self.model.use_ema:
            with self.model.ema_scope():
                self.prepare_model_setting(batch['first_frame'])
                z = self.model.get_input(batch)
                cond = self.model.get_condition(batch)
                # cond = self.replace_cond_latent(cond,self.cond_frames,batch['first_frame'])
                loss,predict = self.model.get_losses(z,cond,return_predict=True)
                self.cond_frames = predict.detach()
                log_prefix = "train" if self.training else "val"
                loss_dict_ema = {f"{log_prefix}/loss":loss}
                self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        else:
            self.prepare_model_setting(batch['first_frame'])
            z = self.model.get_input(batch)
            cond = self.model.get_condition(batch)
            # cond = self.replace_cond_latent(cond,self.cond_frames,batch['first_frame'])
            loss,predict = self.model.get_losses(z,cond,return_predict=True)
            self.cond_frames = predict.detach()
            log_prefix = "train" if self.training else "val"
            loss_dict_no_ema = {f"{log_prefix}/loss":loss}
            self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        


    def configure_optimizers(self):
        lr = self.learning_rate
        params1 = list()
        params2 = list()
        for name,param in self.named_parameters():
            if 'adaptive' in name:
                params1.append(param)
            else:
                if name.startswith('model.model.diffusion_model.input_blocks'):
                    pass
                elif name.startswith('model.model.diffusion_model'):
                    params2.append(param)
        # self.fast_optimizer = torch.optim.AdamW(params1,lr=lr)
        self.slow_optimizer = torch.optim.AdamW(params2,lr=lr)
        if self.use_scheduler:
            assert 'slow' in self.scheduler_config and 'fast' in self.scheduler_config

            scheduler_slow = instantiate_from_config(self.scheduler_config['slow'])
            # scheduler_fast = instantiate_from_config(self.scheduler_config['fast'])

            print("Setting up LambdaLR scheduler...")

            self.schedulers = {"slow":
                {
                    'scheduler':LambdaLR(self.slow_optimizer,lr_lambda=scheduler_slow.schedule),
                    'interval':'step',
                    'frequency':1
                },
                # "fast":
                # {
                #     'scheduler':LambdaLR(self.slow_optimizer,lr_lambda=scheduler_fast.schedule),
                #     'interval':'step',
                #     'frequency':1
                # }
            }
                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Streaming')
    from data.dataloader_streamingSD_sampler import dataloader,DistributedSceneSampler,collate_fn
    parser.add_argument('--config',
                        default='configs/slow_fast_learning.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    data_loader = dataloader(**cfg.data.params.train.params)
    sampler = DistributedSceneSampler(data_loader,samples_per_gpu=2,seed=0)
    # batch_size = 2
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size  =   1,
        num_workers =   0,
        collate_fn=collate_fn,
        sampler=sampler
    )
    network = instantiate_from_config(cfg['model'])
    for _,batch in tqdm(enumerate(data_loader_)):
        network.training_step(batch,_)