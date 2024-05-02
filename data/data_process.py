import torch
import sys
sys.path.append('..')
sys.path.append('.')
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
#from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
#from tools.analysis_tools.visualize.render.base_render import BaseRender
from nuscenes.map_expansion.map_api import NuScenesMap,NuScenesMapExplorer
from pyquaternion import Quaternion
import descartes
from shapely.geometry import Polygon,MultiPolygon,LineString,Point,box
import matplotlib.lines as mlines
import argparse
from nuscenes.nuscenes import NuScenes
import os,random
from torch.utils import data
from utils.tools import get_this_scene_info
import time
import tqdm
import pickle
class DataLoader(data.Dataset):
    def __init__(self,cfg,nusc,spilts,dataset,spilt_name,nusc_maps,device):
        self.spilt_name = spilt_name
        self.ds_dir = cfg.dataset_dir
        self.cfg = cfg
        self.dataset = [dataset[scene_name] for scene_name in spilts[spilt_name]]
        #dataset:dict key:scene_name value:scene
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.device = device
    

    def process(self):
        print("start processing")
        for idx in range(len(self.dataset)):
            print("now_process{}:{}...".format(self.spilt_name,idx))
            out = {'text':[],
               'HDmap':[],
               '3Dbox':[],
               'category':[],
               'reference_image':[],
            }
            scene_sample_token = self.dataset[idx]['first_sample_token']
            scene_token = self.dataset[idx]['token']
            scene_log = self.nusc.get('log',self.dataset[idx]['log_token'])
            nusc_map = self.nusc_maps[scene_log['location']]
            scenes_text = self.dataset[idx]['description']
            out['text'].append(scenes_text)
            count = 0
            pre_translation = None
            while scene_sample_token != '':
                now_reference_image,now_3dbox,now_hdmap,category,yaw,translation = get_this_scene_info(self.cfg,self.nusc,nusc_map,scene_sample_token,count)
                out['3Dbox'].append(np.array(now_3dbox))
                out['reference_image'].append(now_reference_image)
                out['HDmap'].append(now_hdmap)
                out['category'].append(category)
                if idx<1 and count < 5:
                    print('get 3Dbox:{},get hdmap:{}'.format(now_3dbox.shape,now_hdmap.shape))
                count = count + 1
                scene_sample_token = self.nusc.get('sample',scene_sample_token)['next']
                # if pre_translation is None:
                #     out['actions'].append(np.array([yaw,0]))
                # else:
                #     velocity = (translation - pre_translation) 
                #     velocity = np.sqrt(np.sum((velocity**2)))
                #     velocity = velocity / (1 / cfg.camera_frequency)
                #     out['actions'].append(np.array([yaw,velocity]))
                # pre_translation = translation
            with open('./nuScenes/nuScenes_{}_{}'.format(self.spilt_name,idx),'wb') as f:
                pickle.dump(out,f)
        print('finished')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        data_out = self.load_idx(idx)
        data_out['idx'] = torch.from_numpy(np.array(idx,dtype=np.int32))
        return data_out

class Config:
    dataset_dir = "../../datasets/nuScenes"
    camera_frequency = 12

if __name__ == "__main__":
    nusc = NuScenes(version='advanced_12Hz_trainval',dataroot='../../datasets/nuScenes')
    #device = "cuda" if torch.cuda().is_available() else "cpu"
    device = 'cpu'
    spilts = create_splits_scenes()
    scene_dict = {}
    for s in nusc.scene:
        scene_dict[s['name']] = s
    nusc_maps = {
        'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
        'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
        'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
        'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
    }
    cfg = Config()
    data_loader = DataLoader(cfg,nusc,spilts,scene_dict,'train',nusc_maps,device)
    data_loader.process()
    data_loader2 = DataLoader(cfg,nusc,spilts,scene_dict,'val',nusc_maps,device)
    data_loader2.process()


