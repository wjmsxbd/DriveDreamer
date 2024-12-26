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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar,get_this_scene_info_with_lidar_MV,get_global_pose,quaternion_to_matrix,matrix_to_rotation_6d
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from nuscenes.eval.common.utils import quaternion_yaw
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import time
import math
import imageio
try:
    import moxing as mox

    mox.file.shift('os', 'mox')
except:
    pass

from einops import rearrange,repeat
from tqdm import tqdm
import copy
from torch.utils.data import Sampler
from typing import Union,List

def to_tensor(x:list):
    return torch.tensor(x)

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,movie_len,split_name='train',return_pose_info=False,collect_condition=None,use_original_action=True):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        category = []
        for x in self.nusc.category:
            category.append(x['name'])
        print(category)
        self.movie_len = movie_len
        self.num_boxes = num_boxes
        #self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        # nusc_canbus_frequency = cfg['nusc_canbus_frequency']
        camera_frequency = cfg['camera_frequency']
        # ailgn_frequency = math.gcd(nusc_canbus_frequency,camera_frequency)
        # self.nusc_canbus_frequecy = nusc_canbus_frequency // ailgn_frequency
        # self.camera_frequency = camera_frequency // ailgn_frequency
        self.camera_frequency = camera_frequency
        self.num_cameras = cfg['num_cameras']
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        # self.nusc_can = NuScenesCanBus(dataroot='/storage/group/4dvlab/datasets/nuScenes')
        self.return_pose_info = return_pose_info
        self.collect_condition = collect_condition
        self.use_original_action = use_original_action
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
        scenes_id = 0
        scenes = []
        first_frame_idx = []
        first_idx = 0
        for key,value in pic_infos.items():
            value = list(sorted(value,key=lambda e: e['timestamp']))
            sample_steps = 2 // self.camera_frequency if self.cfg['version'] == 'v1.0-mini' else 12 // self.camera_frequency
            assert sample_steps > 0
            first_idx = idx
            frames = torch.arange(len(value)).to(torch.long)[::sample_steps]
            for frame in frames:
                video_infos[idx] = value[frame]
                scenes.append(scenes_id)
                first_frame_idx.append(first_idx)
                idx += 1
            scenes_id += 1
        self.video_infos = video_infos
        self.scenes = np.array(scenes)
        self.first_frame_idx = first_frame_idx

    def __len__(self):
        return len(self.video_infos)
    
    def __getitem__(self,idx):
        if isinstance(idx,list):
            data = []
            # if self.check_idx_is_first_frame(idx[0]):
            #     self.sigmas = self.sigma_sampler(len(idx))
            for i in range(len(idx)):
                data.append(self.get_data_info(idx[i],i))
            return data
        else:
            return self.get_data_info(idx)
    
    def get_cam_image_from_sample_token(self,sample_token,img_size,):
        sample_record = self.nusc.get('sample',sample_token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
        cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
        cam_front_img = mpimg.imread(cam_front_path)
        cam_front_img = Image.fromarray(cam_front_img)
        cam_front_img = cam_front_img.resize(img_size)
        cam_front_img = np.array(cam_front_img)
        cam_front_img = torch.from_numpy(cam_front_img / 255. * 2 - 1.).to(torch.float32)
        cam_front_img = rearrange(cam_front_img,'h w c -> c h w').contiguous()
        return cam_front_img
    
    def get_cam_MVimage_from_sample_token(self,sample_token,img_size,):
        ORI_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
     ]
        cam_MVImage  = []
        for view in ORI_ORDER:
            sample_record = self.nusc.get('sample',sample_token)
            cam_front_token = sample_record['data'][view]
            cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
            cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
            cam_front_img = mpimg.imread(cam_front_path)
            cam_front_img = Image.fromarray(cam_front_img)
            cam_front_img = cam_front_img.resize(img_size)
            cam_front_img = np.array(cam_front_img)
            cam_front_img = torch.from_numpy(cam_front_img / 255. * 2 - 1.).to(torch.float32)
            cam_MVImage.append(cam_front_img)
        cam_MVImage = torch.stack(cam_MVImage, dim=0)
        cam_MVImage = rearrange(cam_MVImage,'n h w c -> n c h w').contiguous()
        return cam_MVImage
        
    def check_idx_is_first_frame(self,idx):
        return idx==0 or self.scenes[idx] != self.scenes[idx-1]

    def get_data_info(self,idx):
        video_info = self.video_infos[idx]
        out = {}
        #out['sigmas'] = self.sigmas[list_idx]
        out['first_frame'] = ([1] if idx == 0 or self.scenes[idx] != self.scenes[idx-1] else [0])
        out['3Dbox'] = []
        if self.num_cameras == 1:
            out['HDmap'] = torch.zeros((3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['image'] = torch.zeros((3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['cond_frames'] = torch.zeros((3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['clip_first_frame'] = torch.zeros((3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
        else:
            out['HDmap'] = torch.zeros((self.num_cameras,3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['image'] = torch.zeros((self.num_cameras,3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['cond_frames'] = torch.zeros((self.num_cameras,3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
            out['clip_first_frame'] = torch.zeros((self.num_cameras,3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
        for i in range(self.movie_len):
            sample_token = video_info['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            text = scene['description']
            log_token = scene['log_token']
            log = self.nusc.get('log',log_token)
            nusc_map = self.nusc_maps[log['location']]
            if self.cfg['img_size'] is not None:
                if self.num_cameras == 1:
                    collect_data = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']),return_camera_info=False,collect_data=self.collect_condition)
                    img = collect_data['reference_image'][:,:,:3].copy()
                    img = torch.from_numpy(img / 255. * 2 - 1.).to(torch.float32)
                    out['image'] = rearrange(img,'h w c -> c h w').contiguous()
                    hdmap = collect_data['HDmap'][:,:,:3].copy()
                    hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
                    out['HDmap'] = rearrange(hdmap,'h w c -> c h w').contiguous()
                else:
                    collect_data = get_this_scene_info_with_lidar_MV(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']),return_camera_info=False,collect_data=self.collect_condition)
                    img = collect_data['reference_image'][:,:,:,:3].copy()
                    img = torch.from_numpy(img / 255. * 2 - 1.).to(torch.float32)
                    out['image'] = rearrange(img,'n h w c -> n c h w').contiguous()
                    hdmap = collect_data['HDmap'][:,:,:,:3].copy()
                    hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
                    out['HDmap'] = rearrange(hdmap,'n h w c -> n c h w').contiguous()
            else:
                if self.num_cameras == 1:
                    collect_data = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']),return_camera_info=False,collect_data=self.collect_condition)
                    img = collect_data['reference_image'][:,:,:3].copy()
                    img = torch.from_numpy(img / 255. * 2 - 1.).to(torch.float32)
                    out['image'] = rearrange(img,'h w c -> c h w').contiguous()
                    hdmap = collect_data['HDmap'][:,:,:3].copy()
                    hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
                    out['HDmap'] = rearrange(hdmap,'h w c -> c h w').contiguous()
                else:
                    collect_data = get_this_scene_info_with_lidar_MV(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']),return_camera_info=False,collect_data=self.collect_condition)
                    img = collect_data['reference_image'][:,:,:,:3].copy()
                    img = torch.from_numpy(img / 255. * 2 - 1.).to(torch.float32)
                    out['image'] = rearrange(img,'n h w c -> n c h w').contiguous()
                    hdmap = collect_data['HDmap'][:,:,:,:3].copy()
                    hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
                    out['HDmap'] = rearrange(hdmap,'n h w c -> n c h w').contiguous()
            boxes = collect_data['3Dbox']
            category = collect_data['category']
            boxes = np.array(boxes).astype(np.float32)
            if boxes.shape[0] == 0:
                box_text = ["None" for i in range(self.num_boxes)]
            elif boxes.shape[0] < self.num_boxes:
                zero_len = self.num_boxes - boxes.shape[0]
                box_text = [f"There is a annotation about {category[i]},the center of callout box is ({np.mean(boxes[i][:8]):.2f},{np.mean(boxes[i][8:]):.2f})" for i in range(boxes.shape[0])]
                for i in range(zero_len):
                    box_text.append('None')
            else:
                boxes = boxes[:self.num_boxes]
                category = category[:self.num_boxes]
                box_text = [f"There is a annotation about {category[i]},the center of callout box is ({np.mean(boxes[i][:8]):.2f},{np.mean(boxes[i][8:]):.2f})" for i in range(boxes.shape[0])] 
            out['3Dbox'] = box_text
        if out['first_frame'][0] == 1:
            out['cond_frames'] = out['image']
        else:
            sample_token = self.video_infos[idx-1]['token']
            if self.num_cameras == 1:
                out['cond_frames'] = self.get_cam_image_from_sample_token(sample_token,tuple(self.cfg['img_size']))
                sample_token = self.video_infos[self.first_frame_idx[idx]]['token']
                out['clip_first_frame'] = self.get_cam_image_from_sample_token(sample_token,tuple(self.cfg['img_size']))
            else :
                out['cond_frames'] = self.get_cam_MVimage_from_sample_token(sample_token,tuple(self.cfg['img_size']))
                sample_token = self.video_infos[self.first_frame_idx[idx]]['token']
                out['clip_first_frame'] = self.get_cam_MVimage_from_sample_token(sample_token,tuple(self.cfg['img_size']))
        return out

def collate_fn(batch):
    out = {}
    batch = batch[0]
    for i in range(len(batch)):
        for key,value in batch[i].items():
            if isinstance(value,torch.Tensor):
                if not key in out.keys():
                    out[key] = value.unsqueeze(0)
                else:
                    out[key] = torch.concat([out[key],value.unsqueeze(0)],dim=0)
            elif isinstance(value,list):
                if not key in out.keys():
                    out[key] = []
                    out[key].append(value)
                else:
                    out[key].append(value)
            elif isinstance(value,dict):
                out[key] = {}
                for k in value.keys():
                    if isinstance(value[k],list):
                        if not k in out[key].keys():
                            out[key][k] = []
                            out[key][k].append(value)
                        else:
                            out[key][k].append(value)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
    return out

class DistributedSceneSampler(Sampler):
    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 train=True,
                 seed=0):
        rank = 3
        world_size = 4 

        print(f"Global rank (RANK): {rank}")
        print(f"World size (WORLD_SIZE): {world_size}")
        _rank,_num_replicas = rank,world_size
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.train = train

        assert hasattr(dataset,'scenes')
        self.scenes = dataset.scenes
        self.max_scene_len = np.bincount(dataset.scenes).max()
        self.len2idx = {}
        pre = 0
        for i in range(1,self.scenes.shape[0]):
            if self.scenes[i] != self.scenes[i-1]:
                scene_len = i - pre
                if not scene_len in self.len2idx.keys():
                    self.len2idx[scene_len] = []
                self.len2idx[scene_len].append(pre)
                pre=i
        if pre != self.scenes.shape[0] - 1:
            scene_len = self.scenes.shape[0] - pre
            if not scene_len in self.len2idx.keys():
                self.len2idx[scene_len] = []
            self.len2idx[scene_len].append(pre)

        self.total_samples = 0
        for scene_len,indices in self.len2idx.items():
            self.total_samples += math.ceil(len(indices) / self.samples_per_gpu)
        self.num_batch = math.ceil(self.total_samples / self.num_replicas)
        for key in self.len2idx.keys():
            self.len2idx[key] = np.array(self.len2idx[key],dtype=np.int32)

        self.up_len = self.num_batch * self.max_scene_len
        print(f"up_len:{self.up_len}")
        
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        #shuffle len
        if self.shuffle:
            scene_lens = torch.randperm(len(self.len2idx.keys())).tolist()
        else:
            scene_lens = torch.arange(0,len(self.len2idx.keys()),1).tolist()
        scene_lens = [list(self.len2idx.keys())[idx] for idx in scene_lens]
        # padding len indices
        len2idx = self.len2idx.copy()
        for scene_len in scene_lens:
            indices = len2idx[scene_len]
            extra = int(math.ceil(indices.shape[0] / self.samples_per_gpu) * self.samples_per_gpu - indices.shape[0])

            if extra:
                if self.shuffle:
                    permutation = torch.randperm(indices.shape[0],generator=g).tolist()
                else:
                    permutation = torch.arange(0,indices.shape[0],1).tolist()
                padding_idx = []
                while extra:
                    if extra < indices.shape[0]:
                        padding_idx.extend([permutation[i] for i in range(extra)])
                        break
                    else:
                        padding_idx.extend([permutation[i] for i in range(indices.shape[0])])
                        extra -= indices.shape[0]
                choose_idx = indices[padding_idx]
                indices = np.concatenate([indices,choose_idx])
            len2idx[scene_len] = indices
            
        # choose interval
        interval_l = self.rank * self.num_batch 
        

        total_samples = 0
        interval_start_len = -1
        i = 0
        while total_samples <= interval_l:
            scene_len = scene_lens[i]
            indices = len2idx[scene_len]
            total_samples += math.ceil(indices.shape[0] / self.samples_per_gpu)
            if total_samples > interval_l:
                interval_start_len = i
                break
            i += 1
            if i == len(scene_lens):
                i = 0
        
        now_batch = 0
        now_scene_len_idx = interval_start_len
        scene_indices = len2idx[scene_lens[now_scene_len_idx]]
        now_idx = scene_indices.shape[0] + (interval_l - total_samples) * self.samples_per_gpu
        assert now_idx >= 0
        indices = []
        while now_batch < self.num_batch * self.samples_per_gpu:
            indices.append([i+scene_indices[now_idx] for i in range(scene_lens[now_scene_len_idx])])
            now_idx += 1
            if now_idx == scene_indices.shape[0]:
                now_scene_len_idx += 1
                if now_scene_len_idx == len(scene_lens):
                    now_scene_len_idx = 0
                now_idx = 0
            scene_indices = len2idx[scene_lens[now_scene_len_idx]]
            now_batch +=1


        iter_indices = []
        for i in range(self.num_batch):
            batch_indices = indices[i*self.samples_per_gpu:(i+1)*self.samples_per_gpu]
            batch_indices = torch.tensor(batch_indices)
            batch_indices = rearrange(batch_indices,'b n -> n b')
            iter_indices.extend(batch_indices.tolist())
        if self.train:
            while len(iter_indices) < self.up_len:
                if len(iter_indices) + scene_lens[now_scene_len_idx] < self.up_len:
                    batch_indices = [[i+scene_indices[now_idx+j] for j in range(self.samples_per_gpu)] for i in range(scene_lens[now_scene_len_idx])]
                    iter_indices.extend(batch_indices)
                else:
                    batch_indices = [[i+scene_indices[now_idx+j] for j in range(self.samples_per_gpu)] for i in range(self.up_len - len(iter_indices))]
                    iter_indices.extend(batch_indices)
                now_idx += self.samples_per_gpu
                if now_idx == scene_indices.shape[0]:
                    now_scene_len_idx += 1
                    if now_scene_len_idx == len(scene_lens):
                        now_scene_len_idx = 0
                    now_idx = 0
                scene_indices = len2idx[scene_lens[now_scene_len_idx]]
            print(len(iter_indices))
            assert len(iter_indices) == self.up_len
        return iter(iter_indices)

    
    def __len__(self):
        if self.train:
            return len(list(self.__iter__()))
        else:
            return self.up_len
    
    def set_epoch(self,epoch):
        self.epoch = epoch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/StreamingSD_cache.yaml',
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
    for _,batch in tqdm(enumerate(data_loader_)):
        pass