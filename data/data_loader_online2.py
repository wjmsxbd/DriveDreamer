import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import torch
import numpy as np
from torch.utils import data
import glob
import pickle

import torch.utils
import torch.utils.data
from omegaconf import DictConfig
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.utils.geometry_utils import view_points,box_in_image
from nuscenes.map_expansion.map_api import NuScenesMap,NuScenesMapExplorer
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from torch.utils import data
from utils.tools import get_this_scene_info
from ldm.util import instantiate_from_config
import os
import matplotlib.image as mpimg
from PIL import Image

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,split_name='train',device=None):
        self.split_name = split_name
        self.cfg = cfg
        self.dataset = glob.glob(cfg['dataroot']+'samples/CAM_FRONT/*.jpg') + glob.glob(cfg['dataroot']+'sweeps/CAM_FRONT/*.jpg')
        self.len = len(self.dataset)
        split = int(self.len * 0.80)
        if split_name == 'train':
            self.dataset = self.dataset[:split]
        else:
            self.dataset = self.dataset[split:]
        self.len = len(self.dataset)
        #self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        # self.device = cfg['device']
        # torch.cuda.set_device(cfg['device'])
        # splits = create_splits_scenes()
        # scene_dict = {}
        # for s in self.nusc.scene:
        #     scene_dict[s['name']] = s
        # self.dataset = []
        # for scene_name in splits[split_name]:
        #     self.dataset.append(scene_dict[scene_name])
        # # self.instantiate_cond_stage(cond_stage_config)
        # self.nusc_maps = {
        #     'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
        #     'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
        #     'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
        #     'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        # }
        # self.len = 165280 if self.split_name == 'train' else 35364

        # self.now_scene_idx = 0
        # self.now_scene_sample_token = None
        # self.log_token = None
        # self.scene_text = None
        

    def instantiate_cond_stage(self,config):
        model = instantiate_from_config(config)
        self.clip = model.eval().cuda()
        self.clip.train = disabled_train
        for param in self.clip.parameters():
            param.requires_grad = False


    def load_idx(self,idx):
        # if self.now_scene_sample_token == None:
        #     self.now_scene_sample_token = self.dataset[self.now_scene_idx]['first_sample_token']
        #     self.scene_log = self.dataset[self.now_scene_idx]['log_token']
        #     self.scene_log = self.nusc.get('log',self.scene_log)
        #     self.scene_text = self.dataset[self.now_scene_idx]['description']
        #     self.nusc_map = self.nusc_maps[self.scene_log['location']]

        # elif self.now_scene_sample_token == '':
        #     self.now_scene_idx += 1
        #     self.now_scene_sample_token = self.dataset[self.now_scene_idx]['first_sample_token']
        #     self.scene_log = self.dataset[self.now_scene_idx]['log_token']
        #     self.scene_log = self.nusc.get('log',self.scene_log)
        #     self.scene_text = self.dataset[self.now_scene_idx]['description']
        #     self.nusc_map = self.nusc_maps[self.scene_log['location']]
        out = {}
        # sample_record = self.nusc.get('sample',self.now_scene_sample_token)
        # cam_front_token = sample_record['data']['CAM_FRONT']
        # cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
        # cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
        cam_front_path = self.dataset[idx]
        cam_front_img = mpimg.imread(cam_front_path)
        #mpimg.imsave(f'./temp/camera_front/{count:02d}.jpg',cam_front_img)
        img_size = (768,448)
        cam_front_img = Image.fromarray(cam_front_img)
        cam_front_img = np.array(cam_front_img.resize(img_size))
        # out['3Dbox'] = np.array(now_3dbox)
        # out['reference_image'] = now_reference_image
        # out['HDmap'] = now_hdmap
        # out['category'] = category
        # out['text'] = self.scene_text

        # out['text'] = self.clip(out['text']).cpu()
        # category_embed = self.clip(out['category']).cpu()
        # category_zero = torch.zeros([self.num_boxes - category_embed.shape[0],category_embed.shape[1]])
        # out['category'] = torch.cat([category_embed,category_zero],dim=0)
        # boxes_zero = np.zeros((self.num_boxes - out['3Dbox'].shape[0],16))
        # out['3Dbox'] = np.concatenate((out['3Dbox'],boxes_zero),axis=0)
        # out['3Dbox'] = torch.from_numpy(out['3Dbox']).to(torch.float32)
        # self.now_scene_sample_token = self.nusc.get("sample",self.now_scene_sample_token)['next']
        out = {}
        # x = torch.randn((448,768,3))
        # x.requires_grad_(True)
        # hdmap = torch.randn((448,768,4))
        # hdmap.requires_grad_(True)
        # text = torch.randn((1,768))
        # text.requires_grad_(True)
        # boxes = torch.randn((50,16))
        # boxes.requires_grad_(True)
        # box_category = torch.randn(50,768)
        # box_category.requires_grad_(True)
        out['reference_image'] = torch.from_numpy(cam_front_img / 255. * 2 - 1.0 ).to(torch.float32)
        # out = {'text':text,
        #     '3Dbox':boxes,
        #     'category':box_category,
        #     'reference_image':x,
        #     'HDmap':hdmap}
        return out

        

    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        data_out = self.load_idx(idx)
        return data_out


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
    cond_stage_config = {}
    config = {}
    config['version'] = 'advanced_12Hz_trainval'
    config['dataroot'] = '../../../share/nuScenes'
    config['device'] = 0
    cond_stage_config['target'] = 'ldm.models.autoencoder.FrozenCLIPTextEmbedder'
    temp = dataloader(config,'train')
    temp.load_idx(0)
    temp.load_idx(1)

        
