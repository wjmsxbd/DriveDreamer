import sys
sys.path.append('.')
sys.path.append('.')
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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar,get_bev_hdmap
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
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

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,movie_len,split_name='train',return_pose_info=False,collect_condition=None):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        self.movie_len = movie_len
        self.num_boxes = num_boxes
        nusc_canbus_frequency = cfg['nusc_canbus_frequency']
        camera_frequency = cfg['camera_frequency']
        ailgn_frequency = math.gcd(nusc_canbus_frequency,camera_frequency)
        self.nusc_canbus_frequecy = nusc_canbus_frequency // ailgn_frequency
        self.camera_frequency = camera_frequency // ailgn_frequency
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        self.nusc_can = NuScenesCanBus(dataroot='/storage/group/4dvlab/datasets/nuScenes')
        self.return_pose_info = return_pose_info
        self.collect_condition = collect_condition
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
        action_infos = {}
        for key,value in pic_infos.items():
            scene_id = int(key[-4:])
            if scene_id in self.nusc_can.can_blacklist:
                continue
            if self.camera_frequency == 1:
                pose = self.nusc_can.get_messages(key,'pose')[::self.nusc_canbus_frequecy]
                value = list(sorted(value,key=lambda e:e['timestamp']))
                frames = torch.arange(len(value)).to(torch.long)[::self.camera_frequency]
                pose_len = len(pose)
                frame_len = len(frames)
                common_len = min(pose_len,frame_len)
                pose = pose[:common_len]
                frames = frames[:common_len]
                chunks = frames.unfold(dimension=0,size=self.movie_len,step=1)
                for ch_id,ch in enumerate(chunks):
                    video_infos[idx] = [value[id] for id in ch]
                    action_infos[idx] = torch.tensor([pose[id]['orientation'] + pose[id]['vel'] for id in ch])
                    idx += 1
            elif self.camera_frequency == 6:
                camera_frequency,nusc_canbus_frequecy = self.camera_frequency * 2,self.nusc_canbus_frequecy * 2
                pose = self.nusc_can.get_messages(key,'pose')
                can_bus_frames = torch.arange(len(pose)).to(torch.float16)
                can_bus_frames = can_bus_frames / nusc_canbus_frequecy
                value = sorted(value,key=lambda e:e['timestamp'])
                camera_frames = torch.arange(len(value)).to(torch.float16) / camera_frequency
                select_can_bus_frames = []
                pos = 0
                for i in range(len(camera_frames)):
                    while pos < len(can_bus_frames) and can_bus_frames[pos] <= camera_frames[i]:
                        pos += 1
                    select_can_bus_frames.append(pos-1)
                frames = torch.arange(len(value))
                chunks = frames.unfold(dimension=0,size=self.movie_len,step=1)
                for ch_id,ch in enumerate(chunks):
                    video_infos[idx] = [value[id] for id in ch]
                    action_infos[idx] = torch.tensor([pose[select_can_bus_frames[id]]['orientation'] + pose[select_can_bus_frames[id]]['vel'] for id in ch])
                    idx+=1
            else:
                raise NotImplementedError
        self.video_infos = video_infos
        self.action_infos = action_infos
        self.save_keys = ['translation','rotation','camera_intrinsic']

    def __len__(self):
        return len(self.video_infos)
    
    def __getitem__(self,idx):
        return self.get_data_info(idx)
    
    def get_data_info(self,idx):
        video_info = self.video_infos[idx]
        actions = self.action_infos[idx]
        out = {}
        out = {}
        out['actions'] = torch.zeros((self.movie_len,14))
        out['sample_points'] = []

        # if self.return_pose_info:
        #     out['cam_front_record'] = {}
        #     out['cam_poserecord'] = {}
        #     out['Lidar_TOP_record'] = {}
        #     out['Lidar_TOP_poserecord'] = {}
        #     init_keys = ['cam_front_record','cam_poserecord','Lidar_TOP_record','Lidar_TOP_poserecord']
        #     for key in init_keys:
        #         for k in self.save_keys:
        #             out[key][k] = [None for i in range(self.movie_len)]
        for i in range(self.movie_len):
            sample_token = video_info[i]['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            text = scene['description']
            log_token = scene['log_token']
            log = self.nusc.get('log',log_token)
            nusc_map = self.nusc_maps[log['location']]
            sample_record = self.nusc.get('sample',sample_token)
            cam_front_token = sample_record['data']['CAM_FRONT']
            sd_record = self.nusc.get('sample_data',cam_front_token)
            pose_record = self.nusc.get('ego_pose',sd_record['ego_pose_token'])
            ego_rotation = torch.tensor(pose_record['rotation']).to(torch.float32)
            ego_translation = torch.tensor(pose_record['translation']).to(torch.float32)
            # cam_front_img,box_list,now_hdmap,box_category,depth_cam_front_img,range_image,dense_range_image
            now_action = torch.cat([actions[i],ego_translation],dim=-1).unsqueeze(0)
            bev_hdmap,sample_points = get_bev_hdmap(cam_front_token,self.nusc,nusc_map,return_point=True)
            if 'actions' in self.collect_condition:
                out['actions'][i] = now_action
            out['sample_points'].append(sample_points)

            
        return out

def collate_fn(batch):
    out = {}
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

def save_tensor_as_image(tensor, file_path,batch_id,batch_size):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.clamp(-1, 1)  # 确保值在[0, 1]之间
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor * 255.0
    tensor = tensor.byte()

    if len(tensor.shape) == 5:
        for idx in range(tensor.shape[0]):
            for frame in range(tensor.shape[1]):
                save_file_path = os.path.join(file_path,f'{batch_id*batch_size+idx:02d}_{frame:02d}.png')
                img = tensor[idx,frame]
                img = img.permute(1,2,0)
                img = img.numpy()
                img = Image.fromarray(img)
                img.save(save_file_path)                
    
def save_tensor_as_video(tensor,file_path,batch_id,batch_size):
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor,-1.,1.)
    if len(tensor.shape) == 5:
        tensor = rearrange(tensor,'b n c h w -> b n h w c')
    else:
        tensor = rearrange(tensor,'b c h w -> b h w c')
    tensor = tensor.numpy()
    tensor = (tensor + 1.0) / 2.0
    tensor = (tensor * 255).astype(np.uint8)
    for i in range(tensor.shape[0]):
        save_video_path = os.path.join(file_path,f'{batch_id*batch_size+i:02d}.gif')
        video = [tensor[i,j] for j in range(tensor.shape[1])]
        imageio.mimsave(save_video_path,video,'GIF',fps=5,loop=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/prediction2_4_dataloader.yaml',
                        type=str,
                        help="config path")
    parser.add_argument('--video',
                        action='store_true',
                        help="use video evaluation")
    parser.add_argument('--train',
                        action='store_true',
                        help="train")
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help="device")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    video_eval = cmd_args.video
    device = cmd_args.device
    use_train = cmd_args.train
    if use_train:
        data_loader = dataloader(**cfg.data.params.train.params)
    else:
        data_loader = dataloader(**cfg.data.params.validation.params)
    batch_size = 2
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size  =   batch_size,
        num_workers =   0,
        collate_fn=collate_fn
    )
    # network = instantiate_from_config(cfg['model'])
    # model_path = 'logs/2024-09-22T17-44-04_svd_2gpus/checkpoints/last.ckpt'
    # network.init_from_ckpt(model_path)
    # network = network.eval().cuda()
    # # # # network = network.eval()
    # save_path = 'all_pics/'
    # save_path = save_path + 'add_adult/'
    # # # # 35
    # cam_real_save_path = save_path + 'cam_inputs/'
    # cam_rec_save_path = save_path + "cam_rec/"
    # cam_samples_save_path = save_path + "cam_samples/"
    # lidar_real_save_path = save_path + 'lidar_inputs/'
    # lidar_rec_save_path = save_path + "lidar_rec/"
    # lidar_samples_save_path = save_path + "lidar_samples/"
    # if not os.path.exists(cam_real_save_path):
    #     os.makedirs(cam_real_save_path)
    # if not os.path.exists(cam_rec_save_path):
    #     os.makedirs(cam_rec_save_path)
    # if not os.path.exists(lidar_real_save_path):
    #     os.makedirs(lidar_real_save_path)
    # if not os.path.exists(lidar_rec_save_path):
    #     os.makedirs(lidar_rec_save_path)
    # if not os.path.exists(cam_samples_save_path):
    #     os.makedirs(cam_samples_save_path)
    # if not os.path.exists(lidar_samples_save_path):
    #     os.makedirs(lidar_samples_save_path)
    # # # l = len(data_loader)

    for _,batch in tqdm(enumerate(data_loader_)):
        pass
        # if device == 'cuda':
        #     batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        # logs = network.log_video(batch)
        # if not video_eval:
        #     save_tensor_as_image(logs['inputs'],file_path=cam_real_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_image(logs['reconstruction'],file_path=cam_rec_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_image(logs['lidar_reconstruction'],file_path=lidar_rec_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_image(logs['lidar_inputs'],file_path=lidar_real_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_image(logs['samples'],file_path=cam_samples_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_image(logs['lidar_samples'],file_path=lidar_samples_save_path,batch_id=_,batch_size=batch_size)
        # else:
        #     save_tensor_as_video(logs['inputs'],file_path=cam_real_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_video(logs['reconstruction'],file_path=cam_rec_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_video(logs['lidar_reconstruction'],file_path=lidar_rec_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_video(logs['lidar_inputs'],file_path=lidar_real_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_video(logs['samples'],file_path=cam_samples_save_path,batch_id=_,batch_size=batch_size)
        #     save_tensor_as_video(logs['lidar_samples'],file_path=lidar_samples_save_path,batch_id=_,batch_size=batch_size)
    #     # batch = {k:v if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    #     # __ = _ + 50
    #     # if __>= l:
    #     #     __ -= l
    #     # swap_data = data_loader.__getitem__(__)
    #     # for key in swap_data.keys():
    #     #     if isinstance(swap_data[key],torch.Tensor):
    #     #         swap_data[key] = swap_data[key].unsqueeze(0)
    #     #     elif isinstance(swap_data[key],list):
    #     #         swap_data[key] = [swap_data[key]]
    #     # swap_data['reference_image'] = batch['reference_image']

    #     # batch = swap_data
    #     batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    #     logs = network.log_video(batch)
    #     save_tensor_as_video(logs['inputs'],file_path=save_path+f'inputs_{_:02d}.png')
    #     save_tensor_as_video(logs['samples'],file_path=save_path+f'samples{_:02d}.png')
    #     save_tensor_as_video(logs['lidar_samples'],file_path=save_path+f'lidar_samples{_:02d}.png')
    #     save_tensor_as_video(logs['lidar_inputs'],file_path=save_path+f'lidar_inputs{_:02d}.png')
    #     pass