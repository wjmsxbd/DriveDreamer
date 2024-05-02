import torch
import numpy as np
from torch.utils import data
import glob
import pickle

import torch.utils
import torch.utils.data
from omegaconf import DictConfig

class dataloader(data.Dataset):
    def __init__(self,cfg,split_name='train'):
        self.split_name = split_name
        self.cfg = cfg
        self.ds_path = cfg.ds_path
        self.datasets = glob.glob(self.ds_path + f'/nuScenes_{self.split_name}*.pkl')
        self.now_read_idx = 0
        self.now_read_data_idx = None
        self.len = 35364
        self.now_read_data = None
        self.batch_size = cfg.batch_size
        max_len = 0
        for dataset in self.datasets:
            with open(dataset,'rb') as f:
                loaded_data = pickle.load(f)
                nb = [loaded_data['3Dbox'][i].shape[0] for i in range(0,len(loaded_data['3Dbox']))]
                max_len = max(max_len,max(nb))
        print(f"all_box_nb_max_{split_name}:{max_len}")

        
        
    def load_idx(self,idx):
        if self.now_read_data == None:
            with open(self.datasets[self.now_read_idx],'rb') as f:
                self.now_read_data = pickle.load(f)
            self.now_read_data_idx = 0
        elif len(self.now_read_data['HDmap']) == self.now_read_data_idx :
            self.now_read_idx += 1
            with open(self.datasets[self.now_read_idx],'rb') as f:
                self.now_read_data = pickle.load(f)
            self.now_read_data_idx = 0
        out = {}
        for key in self.now_read_data.keys():
            if key == 'text':
                out[key] = self.now_read_data[key]
            else:
                out[key] = self.now_read_data[key][self.now_read_data_idx]
        self.now_read_data_idx += 1

        return out

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_out = self.load_idx(idx)
    

class Config:
    batch_size : int = 16
    ds_path : str = './nuScenes'

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}

def build_loader(dataset:data.Dataset,
                cfg: DictConfig,
                split: str = 'train') -> torch.utils.data.DataLoader:
    dataset_cfg = cfg
    is_train = 'train' in split
    is_test  = 'test' in split

    num_workers = dataset_cfg.get('num_workers',DEFAULT_NUM_WORKERS)
    shuffle     = dataset_cfg.get('shuffle', True)
    collate_fn  = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   dataset_cfg.batch_size if not is_test else 1,
        num_workers =   num_workers.get(split, 0),
        collate_fn  =   collate_fn,
        drop_last   =   True and (is_train or not is_test),
        pin_memory  =   dataset_cfg.get('pin_memory', False),
        shuffle     =   shuffle and is_train and not is_test,
    )
    return data_loader

if __name__ == '__main__':
    cfg = Config()
    temp = dataloader(cfg,'val')
    # for i in range(0,len(temp)):
    #     temp.load_idx(i)
    #     print(f'finish:{i}')
    # ds = temp.load_idx(0)
    # print(ds)
    # data_loader = torch.utils.data.DataLoader(temp,cfg.batch_size)
    # for it,batch in enumerate(data_loader):
    #     print(batch)


        
