import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import pickle
import torch
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import FrozenCLIPTextEmbedder
from omegaconf import DictConfig

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

device = "cuda" if torch.cuda.is_available() else 'cpu'

class LSUNBase(Dataset):
    def __init__(self,
                 data_path,
                 num_boxes,
                 cond_stage_config,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 split_name='train'):
        self.data_path = data_path
        self.split_name = split_name
        self.num_boxes = num_boxes
        self.datasets = glob.glob(self.data_path + f'/nuScenes_{self.split_name}*.pkl')
        self.instantiate_cond_stage(cond_stage_config)
        self.now_read_idx = 0
        self.now_read_data_idx = None
        self.len = 0
        self.now_read_data = None
        if self.split_name == 'train':
            self.len = 165280
        else:
            self.len = 35364
    
    def instantiate_cond_stage(self,config):
        model = instantiate_from_config(config)
        self.clip = model.eval().to(device)
        #self.clip = FrozenCLIPTextEmbedder().to(device)
        self.clip.train = disabled_train
        for param in self.clip.parameters():
            param.requires_grad = False

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
                out[key] = self.clip(self.now_read_data[key])
            elif key == '3Dbox':
                out[key] = self.now_read_data[key][self.now_read_data_idx]
                box_zeros = np.zeros((self.num_boxes - out[key].shape[0],16))
                out[key] = np.concatenate((out[key],box_zeros),axis=0)
                out[key] = torch.from_numpy(out[key]).to(torch.float32)
            elif key == 'category':
                category = self.now_read_data[key][self.now_read_data_idx]
                category_embed = self.clip(category)
                out[key] = category_embed.cpu()
                category_zero = torch.zeros([self.num_boxes - category_embed.shape[0],category_embed.shape[1]])
                out[key] = torch.cat([out[key],category_zero],dim=0)
                # print("!!!!!!get clip")
                # print(f"clip_shape:{category_embed.shape} device:{category_embed.device}")
                
            else:
                out[key] = self.now_read_data[key][self.now_read_data_idx]
                assert out[key] is not None, f"{key} got none."
        self.now_read_data_idx += 1
        sh = [len(out[key]) if isinstance(out[key],list) else out[key].shape for key in out.keys()]
        print(sh)
        return out
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_out = self.load_idx(idx)
        return data_out

# class Config:
#     target = 'ldm.models.autoencoder.FrozenCLIPTextEmbedder'

class LSUNChurchesTrain(LSUNBase):
    def __init__(self,**kwargs):
        super().__init__(data_path='./nuScenes',split_name='train',**kwargs)

class LSUNChurchesValidation(LSUNBase):
    def __init__(self,**kwargs):
        super().__init__(data_path='./nuScenes',split_name='val',**kwargs)

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}

def build_dataloader(dataset: torch.utils.data.Dataset,
                     cfg: DictConfig,
                     split: str = 'train',) -> torch.utils.data.DataLoader:
    dataset_cfg = cfg
    is_train    = 'train' in split
    is_test    = 'test' in split

    num_workers = dataset_cfg['num_workers'][split]
    shuffle     = dataset_cfg['shuffle'][split]

    collate_fn  = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   dataset_cfg['batch_size'][split] if not is_test else 1,
        num_workers =   num_workers,
        collate_fn  =   collate_fn,
        drop_last   =   True and (is_train or not is_test),
        pin_memory  =   dataset_cfg.get('pin_memory', False),
        shuffle     =   shuffle and is_train and not is_test,
    )
    return data_loader

# if __name__ == '__main__':
#     tmp = LSUNChurchesValidation()

#     tmp.load_idx(0)
