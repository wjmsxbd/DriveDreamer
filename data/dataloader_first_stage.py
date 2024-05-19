import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import torch
import numpy as np
from torch.utils import data
import glob
import pickle
import os
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
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,cond_stage_config,split_name='train'):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        self.device = cfg['device']
        self.num_boxes = num_boxes
        torch.cuda.set_device(cfg['device'])
        self.instantiate_cond_stage(cond_stage_config)
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        self.instantiate_cond_stage(cond_stage_config)
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        data_infos = pickle.load(data_info_path)
        self.data_infos = list(sorted(data_infos['infos'],key=lambda e:e['timestamp']))

    def __len__(self):
        return len(self.data_infos)
    
    def instantiate_cond_stage(self,config):
        model = instantiate_from_config(config)
        self.clip = model.eval().cuda()
        self.clip.train = disabled_train
        for param in self.clip.parameters():
            param.requires_grad = False


    def get_data_info(self,idx):
        sample_token = self.data_infos[idx]['token']
        # sample_record = self.nusc.get('sample',sample_token)
        # cam_front_token = sample_record['data']['CAM_FRONT']
        # cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
        # cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
        # cam_front_img = mpimg.imread(cam_front_path)
        # imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
        # cam_front_img = Image.fromarray(cam_front_img)
        ref_img,boxes,hdmap,category,yaw,translation = get_this_scene_info(self.cfg['dataroot'],self.nusc,self.nusc_maps,sample_token)
        out = {}
        boxes = np.array(boxes).astype(np.float32)
        out['reference_image'] = ref_img
        out['HDmap'] = hdmap
        out['text'] = self.clip(out['text']).cpu()
        if boxes.shape[0] == 0:
            boxes = np.zeros((self.num_boxes,16))
            category = np.zeros((self.num_boxes,out['text'].shape[1]))
        else:
            boxes_zero = np.zeros((self.num_boxes - boxes.shape[0],16))
            boxes = np.concatenate((boxes,boxes_zero),axis=0)
            category_embed = self.clip(out['category']).cpu()
            category_zero = torch.zeros([self.num_boxes-category_embed.shape[0],category_embed.shape[1]])
            category = torch.cat([category_embed,category_zero],dim=0)
        out['3Dbox'] = boxes
        out['category'] = category
        return out

    def __getitem__(self,idx):
        return self.get_data_info(idx)
    

if __name__ == '__main__':
    pass
