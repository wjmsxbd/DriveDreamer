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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import time
import math

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
            'boston-seaport': NuScenesMap(dataroot=cfg['dataroot'], map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-queenstown'),
        }
        self.nusc_can = NuScenesCanBus(dataroot=cfg['dataroot'])
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
        for key in self.collect_condition:
            if key == '3Dbox':
                out[key] = torch.zeros((self.movie_len,self.num_boxes,16))
                out['category'] = [None for i in range(self.movie_len)]
            elif key == 'text':
                out[key] = [None for i in range(self.movie_len)]
            elif key == 'actions':
                out[key] = torch.zeros((self.movie_len,14))
            else:
                out[key] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))

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
            # cam_front_img,box_list,now_hdmap,box_category,depth_cam_front_img,range_image,dense_range_image
            if self.cfg['img_size'] is not None:
                collect_data = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']),return_camera_info=False,collect_data=self.collect_condition)
            else:
                collect_data = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,return_camera_info=False,collect_data=self.collect_condition)
            
            cam_front_translation = torch.tensor(collect_data['cam_front_record']['translation'])
            cam_front_rotation = torch.tensor(collect_data['cam_front_record']['rotation'])
            now_action = torch.cat([actions[i],cam_front_translation,cam_front_rotation],dim=-1).unsqueeze(0)
            

            if 'text' in self.collect_condition:
                out['text'][i] = text
            if 'actions' in self.collect_condition:
                out['actions'][i] = now_action
            for key in collect_data:
                if key == '3Dbox':
                    boxes = collect_data['3Dbox']
                    category = collect_data['category']
                    boxes = np.array(boxes).astype(np.float32)
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
                        boxes = torch.from_numpy(copy.deepcopy(boxes[:self.num_boxes]))
                        category_embed = copy.deepcopy(category[:self.num_boxes])
                        category = copy.deepcopy(category_embed[:self.num_boxes])
                    out['3Dbox'][i] = boxes
                    out['category'][i] = category
                elif key == 'category' or key == 'actions' or key == 'cam_front_record':
                    pass
                else:
                    img = collect_data[key]
                    img = img[:,:,:3].copy()
                    img = torch.from_numpy(img / 255. * 2 - 1.).to(torch.float32)
                    out[key][i] = img
            # if self.return_pose_info:
            #     for key in self.save_keys:
            #         if key in cam_front_record.keys():
            #             out['cam_front_record'][key][i] = cam_front_record[key]
            #     for key in self.save_keys:
            #         if key in cam_poserecord.keys():
            #             out['cam_poserecord'][key][i] = cam_poserecord[key]
            #     for key in self.save_keys:
            #         if key in Lidar_TOP_record.keys():
            #             out['Lidar_TOP_record'][key][i] = Lidar_TOP_record[key]
            #     for key in self.save_keys:
            #         if key in Lidar_TOP_poserecord.keys():
            #             out['Lidar_TOP_poserecord'][key][i] = Lidar_TOP_poserecord[key]
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

def save_tensor_as_image(tensor, file_path):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)

    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # tensor = tensor * std[:, None, None] + mean[:, None, None]

    tensor = tensor.clamp(-1, 1)  # 确保值在[0, 1]之间
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor * 255.0

    tensor = tensor.byte()


    tensor = tensor.permute(1, 2, 0)


    numpy_array = tensor.numpy()

    image = Image.fromarray(numpy_array)

    image.save(file_path)

def save_tensor_as_video(tensor,file_path):
    import imageio
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor,-1.,1.)
    filename = file_path
    tensor = rearrange(tensor,'b n c h w -> b n h w c')
    tensor = tensor.numpy()
    tensor = (tensor + 1.0) / 2.0
    tensor = (tensor * 255).astype(np.uint8)
    writer = imageio.get_writer(filename,fps=5)
    for i in range(tensor.shape[1]):
        writer.append_data(tensor[:,i])
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/prediction2_4.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)

    data_loader = dataloader(**cfg.data.params.train.params)
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size  =   1,
        num_workers =   2,
        collate_fn=collate_fn
    )
    # network = instantiate_from_config(cfg['model'])
    # model_path = 'logs/2024-08-22T20-20-05_prediction2_4/checkpoints/epoch=000390.ckpt'
    # network.init_from_ckpt(model_path)
    # network = network.eval().cuda()
    # # network = network.eval()
    # save_path = 'all_pics/'
    # save_path = save_path + 'prediction2_4/'
    # 35
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    for _,batch in tqdm(enumerate(data_loader_)):
        # batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        # # batch = {k:v if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        # logs = network.log_video(batch)
        # save_tensor_as_video(logs['inputs'],file_path=save_path+f'inputs_{_:02d}.png')
        # save_tensor_as_video(logs['samples'],file_path=save_path+f'samples{_:02d}.png')
        # save_tensor_as_video(logs['lidar_samples'],file_path=save_path+f'lidar_samples{_:02d}.png')
        # save_tensor_as_video(logs['lidar_inputs'],file_path=save_path+f'lidar_inputs{_:02d}.png')
        pass