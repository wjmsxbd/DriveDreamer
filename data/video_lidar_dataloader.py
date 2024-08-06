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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
import time

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,movie_len,split_name='train'):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        self.movie_len = movie_len
        self.num_boxes = num_boxes
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        with open(data_info_path,'rb') as f:
            data_infos = pickle.load(f)
        data_infos = data_infos['infos']
        print(f"len:{len(data_infos)}")
        pic_infos = {}
        video_infos = {}
        for id in range(len(data_infos)):
            sample_token = data_infos[id]['token']
            scene_token = self.nusc.get("sample",sample_token)['scene_token']
            scene = self.nusc.get("scene",scene_token)
            if not scene['name'] in pic_infos.keys():
                pic_infos[scene['name']] = [data_infos[id]]
            else:
                pic_infos[scene['name']].append(data_infos[id])
        idx = 0
        for key,value in pic_infos.items():
            value = list(sorted(value,key=lambda e:e['timestamp']))
            frames = torch.arange(len(value)).to(torch.long)
            chunks = frames.unfold(dimension=0,size=self.movie_len,step=1)
            for ch_id,ch in enumerate(chunks):
                video_infos[idx] = [value[id] for id in ch]
                idx += 1
        self.video_infos = video_infos
    
    def __len__(self):
        return len(self.video_infos)
    def get_data_info(self,idx):
        video_info = self.video_infos[idx]
        out = {}
        for i in range(self.movie_len):
            sample_token = video_info[i]['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            text = scene['description']
            log_token = scene['log_token']
            log = self.nusc.get('log',log_token)
            nusc_map = self.nusc_maps[log['location']]
            # cam_front_img,box_list,now_hdmap,box_category,depth_cam_front_img,range_image,dense_range_image
            if self.cfg['img_size'] is not None:
                ref_img,boxes,hdmap,category,depth_cam_front_img,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']))
            else:
                ref_img,boxes,hdmap,category,depth_cam_front_img,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token)
            dense_range_image = np.repeat(dense_range_image[...,np.newaxis],3,axis=-1)

            boxes = np.array(boxes).astype(np.float32)
            ref_img = torch.from_numpy(ref_img / 255. * 2 - 1.).to(torch.float32)
            hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
            range_image = torch.from_numpy(range_image / 255. * 2 - 1.).to(torch.float32)
            dense_range_image = torch.from_numpy(dense_range_image / 255. * 2 - 1.).to(torch.float32)
            depth_cam_front_img = torch.from_numpy(depth_cam_front_img / 255. * 2 - 1.).to(torch.float32)

            if not 'reference_image' in out.keys():
                out['reference_image'] = ref_img.unsqueeze(0)
            else:
                out['reference_image'] = torch.cat([out['reference_image'],ref_img.unsqueeze(0)],dim=0)

            if not "HDmap" in out.keys():
                out['HDmap'] = hdmap[:,:,:3].unsqueeze(0)
            else:
                out['HDmap'] = torch.cat([out['HDmap'],hdmap[:,:,:3].unsqueeze(0)],dim=0)

            if not "range_image" in out.keys():
                out['range_image'] = range_image.unsqueeze(0)
            else:
                out['range_image'] = torch.cat([out['range_image'],range_image.unsqueeze(0)],dim=0)

            if not "dense_range_image" in out.keys():
                out['dense_range_image'] = dense_range_image.unsqueeze(0)
            else:
                out['dense_range_image'] = torch.cat([out['dense_range_image'],dense_range_image.unsqueeze(0)],dim=0)

            if not 'depth_cam_front_img' in out.keys():
                out['depth_cam_front_img'] = depth_cam_front_img.unsqueeze(0)
            else:
                out['depth_cam_front_img'] = torch.cat([out['depth_cam_front_img'],depth_cam_front_img.unsqueeze(0)],dim=0)

            if not 'text' in out.keys():
                out['text'] = [text]
            else:
                out['text'] = out['text'] + [text]

            if boxes.shape[0] == 0:
                boxes = torch.from_numpy(np.zeros((self.num_boxes,16)))
                category = ["None" for i in range(self.num_boxes)]
            elif boxes.shape[0] < self.num_boxes:
                zero_len = self.num_boxes - boxes.shape[0]
                boxes_zero = np.zeros((self.num_boxes-boxes.shape[0],16))
                boxes = torch.from_numpy(np.concatenate((boxes,boxes_zero),axis=0))
                category_none = ["None" for i in range(zero_len)]
                category = category + category_none
            else:
                boxes = torch.from_numpy(boxes[:self.num_boxes])
                category_embed = category[:self.num_boxes]
                category = category_embed[:self.num_boxes]

            if not '3Dbox' in out.keys():
                out['3Dbox'] = boxes.unsqueeze(0).to(torch.float32)
                out['category'] = [category]
            else:
                out['3Dbox'] = torch.cat([out['3Dbox'],boxes.unsqueeze(0)],dim=0)
                out['category'] = out['category'] + [category]
        return out
    def __getitem__(self,idx):
        return self.get_data_info(idx)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/global_condition.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)

    data_loader = dataloader(**cfg.data.params.train.params)
    # 35
    out = data_loader.__getitem__(35)