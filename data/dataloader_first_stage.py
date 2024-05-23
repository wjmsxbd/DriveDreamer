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
from einops import repeat
import omegaconf

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,cond_stage_config,split_name='train',device=None):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        if device is not None:
            self.device = f'cuda:{device}'
        self.num_boxes = num_boxes
        #torch.cuda.set_device(cfg['device'])
        self.instantiate_cond_stage(cond_stage_config)
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        #$self.instantiate_cond_stage(cond_stage_config)
        self.clip = self.clip.to(self.device)
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        with open(data_info_path,'rb') as f:
            data_infos = pickle.load(f)
        self.data_infos = list(sorted(data_infos['infos'],key=lambda e:e['timestamp']))

    def __len__(self):
        return len(self.data_infos)
    
    def instantiate_cond_stage(self,config):
        model = instantiate_from_config(config)
        self.clip = model.eval()
        for param in self.clip.parameters():
            param.requires_grad = False


    def get_data_info(self,idx):
        sample_token = self.data_infos[idx]['token']
        scene_token = self.nusc.get('sample',sample_token)['scene_token']
        scene = self.nusc.get('scene',scene_token)
        text = scene['description']
        print(f'get:text {text}')
        log_token = scene['log_token']
        log = self.nusc.get('log',log_token)
        nusc_map = self.nusc_maps[log['location']]
        

        # sample_record = self.nusc.get('sample',sample_token)
        # cam_front_token = sample_record['data']['CAM_FRONT']
        # cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
        # cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
        # cam_front_img = mpimg.imread(cam_front_path)
        # imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
        # cam_front_img = Image.fromarray(cam_front_img)
        ref_img,boxes,hdmap,category,yaw,translation = get_this_scene_info(self.cfg['dataroot'],self.nusc,nusc_map,sample_token)
        out = {}
        boxes = np.array(boxes).astype(np.float32)
        ref_img = torch.from_numpy(ref_img / 255.).to(torch.float32)
        hdmap = torch.from_numpy(hdmap / 255.).to(torch.float32)

        out['reference_image'] = ref_img.unsqueeze(0)
        out['HDmap'] = hdmap.unsqueeze(0)
        out['text'] = self.clip(text).cpu().to(torch.float32)
        out['text'] = repeat(out['text'],'n c -> (repeat n) c',repeat=out['reference_image'].shape[0])
        if boxes.shape[0] == 0:
            boxes = torch.from_numpy(np.zeros((self.num_boxes,16)))
            category = torch.from_numpy(np.zeros((self.num_boxes,out['text'].shape[1])))
        elif boxes.shape[0]<self.num_boxes:
            boxes_zero = np.zeros((self.num_boxes - boxes.shape[0],16))
            boxes = torch.from_numpy(np.concatenate((boxes,boxes_zero),axis=0))
            category_embed = self.clip(category).cpu()
            category_zero = torch.zeros([self.num_boxes-category_embed.shape[0],category_embed.shape[1]])
            category = torch.cat([category_embed,category_zero],dim=0)
        else:
            boxes = torch.from_numpy(boxes[:self.num_boxes])
            category_embed = self.clip(category).cpu()
            category = category_embed[:self.num_boxes]
        out['3Dbox'] = boxes.unsqueeze(0).to(torch.float32)
        out['category'] = category.unsqueeze(0).to(torch.float32)
        return out

    def __getitem__(self,idx):
        return self.get_data_info(idx)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/first_stage_step1_config_online2.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    print("get cfg!!!!!!!!!!!!")
    data_loader = dataloader(**cfg.data.params.train.params)
    out = data_loader.__getitem__(0)
    print([v.shape for k,v in out.items()])
    print(out.keys())
    

