import sys
sys.path.append('.')
sys.path.append('..')
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from utils.tools import makepath,LOGGER_DEFAULT_FORMAT
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from torch.distributed import init_process_group,destroy_process_group
from ldm.models.diffusion.ddpm import AutoDM
import shutil
from ldm.data.lsun import build_dataloader
from datetime import datetime
from tensorboardX import SummaryWriter
torch.set_default_dtype(torch.float32)
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

class Trainer(nn.Module):
    def __init__(self,cfg,inference=False):
        super().__init__()
        self.dtype = torch.float32
        self.cfg = cfg
        self.is_inference = inference
        torch.manual_seed(cfg.seed)

        start_time = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir)
        logger_path = makepath(os.path.join(cfg['work_dir'],'%s_%s.log' %(cfg['expr_ID'],'train' if not inference else 'test')),isfile=True)
        logger.add(logger_path,backtrace=True,diagnose=True)
        logger.add(lambda x:x,
                   level=cfg['logger_level'].upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info
        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        self.logger('[%s] - Started training XXX, experiment code %s' % (cfg['expr_ID'], start_time))
        self.logger('tensorboard --logdir=%s' % summary_logdir)
        self.logger('Torch Version: %s\n' % torch.__version__)
        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0], os.path.join(cfg['work_dir'], os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(stime, '%Y%m%d_%H%M'))))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_completed = 0
        self.load_data(cfg)
        #self.loss_setup()
        self.network = instantiate_from_config(cfg['model']).to(self.device)
        self.configure_optimizers()
        self.best_loss = np.inf

        if cfg['num_gpus'] > 1:
            os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICE'] = ','.join(cfg['device_ids'])
            self.network = nn.DataParallel(self.network,device_ids=cfg['device_ids'])

        self.restarted_from_ckpt = False

        if inference and cfg['best_model'] is None:
            cfg['best_model'] = sorted(glob.glob(os.path.join(cfg['work_dir'], 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        if cfg['best_model'] is not None:
            self._get_network().load_state_dict(torch.load(cfg['best_model'], map_location=self.device), strict=False)
            self.logger('Restored trained model from %s' % cfg['best_model'])
            self.restarted_from_ckpt = True

    def load_data(self,cfg):
        self.ds_train = instantiate_from_config(cfg['data']['train'])
        self.ds_val = instantiate_from_config(cfg['data']['val'])
        self.ds_train = build_dataloader(self.ds_train,cfg['data'],'train')
        self.ds_val = build_dataloader(self.ds_val,cfg['data'],'val')
        self.logger('Dataset Train, Vald size respectively: %.2f M, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3))
        
    def _get_network(self):
        return self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network
    
    def save_network(self):
        torch.save(self.network.module.state_dict()
                   if isinstance(self.network, torch.nn.DataParallel)
                   else self.network.state_dict(), self.cfg['best_model'])

    def forward(self,x):
        return self.network(x)

    def configure_optimizers(self):
        opt,scheduler = self.network.configure_optimizers()
        self.opt = opt[0]
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(**self.cfg['early_stopping'], trace_func=self.logger)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)

    def train(self):
        self.network.train()
        save_every_it = len(self.ds_train) / self.cfg['summary_steps']
        train_loss_dict = {}
        for it,batch in enumerate(self.ds_train):
            self.network.on_train_batch_start(batch,it,self.epochs_completed)
            self.opt.zero_grad()
            loss,loss_dict = self.network.shared_step(batch)
            
            loss.backward()
            self.opt.step()
            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in loss_dict.items()}
            if it % (save_every_it+1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='MNet',
                                                    it=it,
                                                    try_num=0,
                                                    mode='train')

                self.logger(train_msg)
        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}

        return train_loss_dict

    def evaluate(self):
        self.network.eval()
        eval_loss_dict = {}
        data = self.ds_val

        with torch.no_grad():
            for it,batch in enumerate(self.ds_val):
                self.opt.zero_grad()
                loss,loss_dict = self.network.share_step(batch)
                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in loss_dict.items()}

            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def fit(self,n_epochs=None,message=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf

        for epoch_num in range(1,n_epochs+1):
            self.logger('--- starting Epoch # %03d' % epoch_num)
            train_loss_dict = self.train()
            eval_loss_dict  = self.evaluate()

            self.lr_scheduler.step(epoch_num)

            cur_lr = self.lr_scheduler.lr_lambdas
            if cur_lr != prev_lr:
                self.logger('--- learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            with torch.no_grad():
                eval_msg = Trainer.create_loss_message(eval_loss_dict, expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                        model_name='AutoDM',
                                                        try_num=0, mode='evald')
                if eval_loss_dict[''] < self.best_loss:
                    self.cfg.best_model = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model.pt' % (self.epochs_completed)), isfile=True)
                    self.save_network()
                    self.logger(eval_msg + ' ** ')
                    self.best_loss = eval_loss_dict['']
                else:
                    self.logger(eval_msg)
                
                #TODO:write loss
                # self.swriter.add_scalars('total_loss/scalars',
                #                          {'train_loss_total': train_loss_dict['loss_total'],
                #                          'evald_loss_total': eval_loss_dict['loss_total'], },
                #                          self.epochs_completed)
            
            if self.early_stopping(eval_loss_dict['']):
                self.logger('Early stopping the training!')
                break
            
            self.epochs_completed += 1
        endtime = datetime.now().replace(microsecond=0)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss))
        self.logger('Best model path: %s\n' % self.cfg.best_model)

            



def train():
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/first_stage_step1_config2.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = OmegaConf.load(cmd_args.config)
    makepath(cfg['work_dir'])
    run_trainer_once(cfg)

def run_trainer_once(cfg):
    trainer = Trainer(cfg)
    OmegaConf.save(trainer.cfg,os.path.join(cfg.work_dir,"{}.yaml".format(cfg.expr_ID)))    
    trainer.fit()
    OmegaConf.save(trainer.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))

if __name__ == '__main__':
    mp.set_start_method('spawn')
    train()