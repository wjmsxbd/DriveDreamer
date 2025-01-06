import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import scipy.ndimage
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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar,get_global_pose,quaternion_to_matrix,matrix_to_rotation_6d,get_bev_box_label,get_bev_hdmap_front_view
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
import scipy

def to_tensor(x:list):
    return torch.tensor(x)
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
        self.observe_category = ['background','human','vehicle']
        self.instance_label = {}
        instance_id = 0
        for category in self.observe_category:
            self.instance_label[category] = instance_id
            instance_id += 1
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        with open(data_info_path,'rb') as f:
            data_infos = pickle.load(f)
        data_infos = data_infos['infos']
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
                    action_infos[idx] = torch.vstack([torch.cat([to_tensor(pose[id]['vel']),to_tensor(pose[id]['accel']),matrix_to_rotation_6d(quaternion_to_matrix(to_tensor(pose[id]['orientation'])))],dim=-1) for id in ch])
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
                    while pos < len(can_bus_frames) and can_bus_frames[pose] <= camera_frames[i]:
                        pos += 1
                    select_can_bus_frames.append(pos-1)
                frames = torch.arange(len(value))
                chunks = frames.unfold(dimension=0,size=self.movie_len,step=1)
                for ch_id,ch in enumerate(chunks):
                    video_infos[idx] = [value[id] for id in ch]
                    action_infos[idx] = torch.vstack([torch.cat([to_tensor(pose[select_can_bus_frames[id]]['vel']),to_tensor(pose[select_can_bus_frames[id]]['accel']),matrix_to_rotation_6d(quaternion_to_matrix(to_tensor(pose[select_can_bus_frames[id]]['orientation'])))],dim=-1) for id in ch])
                    idx += 1
            else:
                raise NotImplementedError
        self.video_infos = video_infos
        self.action_infos = action_infos
        
    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self,idx):
        return self.get_data_info(idx)
    
    def get_cam_image_from_sample_token(self,sample_token,img_size,):
        sample_record = self.nusc.get('sample',sample_token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        cam = self.nusc.get('sample_data',cam_front_token)
        cs_record = self.nusc.get('calibrated_sensor',cam['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])

        imsize = (cam['width'],cam['height'])

        cam_front_path = cam['filename']
        cam_front_path = os.path.join(self.cfg['dataroot'],cam_front_path)
        cam_front_img = mpimg.imread(cam_front_path)
        cam_front_img = Image.fromarray(cam_front_img)
        cam_front_img = cam_front_img.resize(img_size)
        cam_front_img = np.array(cam_front_img)
        cam_front_img = torch.from_numpy(cam_front_img).to(torch.float32)
        cam_front_img = rearrange(cam_front_img,'h w c -> c h w').contiguous()
        camera_intrinsic = torch.from_numpy(camera_intrinsic).to(torch.float32)
        camera_intrinsic[0] = (img_size[0] / imsize[0]) * camera_intrinsic[0]
        camera_intrinsic[1] = (img_size[1] / imsize[1]) * camera_intrinsic[1]
        return cam_front_img,camera_intrinsic

    def get_cam2ego_matrix(self,sample_token):
        sample_record = self.nusc.get('sample',sample_token)
        cam_front_token = sample_record['data']['CAM_FRONT']
        cam = self.nusc.get('sample_data',cam_front_token)
        cs_record = self.nusc.get('calibrated_sensor',cam['calibrated_sensor_token'])
        cam2ego = torch.zeros((4,4)).to(torch.float32)
        cam2ego[:3,:3] = torch.from_numpy(Quaternion(np.array(cs_record['rotation'])).rotation_matrix.reshape(3,3))
        cam2ego[:3,3:] = torch.from_numpy(np.array(cs_record['translation']).reshape(3,1))
        return cam2ego

    def calculate_birdview_labels(self,bev_hdmap,bev_box):
        birdview = torch.cat([bev_box,bev_hdmap],dim=0)
        # birdview = bev_box
        birdview_labels = torch.argmax(birdview,dim=0).to(torch.long)
        birdview_labels = birdview_labels[None]
        return birdview_labels
    
    def get_map_label(self,bev_hdmap):
        background_mask = (bev_hdmap == 0)
        background = torch.ones_like(background_mask) * background_mask * 127
        bev_hdmap = torch.cat([background,bev_hdmap],dim=0)
        map_label = torch.argmax(bev_hdmap,dim=0).to(torch.long)
        map_label = map_label[None]
        return map_label
    
    def crop_image(self,image,center_x,center_y,size):
        x_min = int(center_x - size)
        x_max = int(center_x + size)
        y_min = int(center_y - size * 2)
        y_max = int(center_y )
        crop_image = image[:,y_min:y_max,x_min:x_max].copy()
        return crop_image

    def get_data_info(self,idx):
        video_info = self.video_infos[idx]
        actions = self.action_infos[idx]
        out = {}
        out['vel'] = torch.zeros((self.movie_len,3))
        out['accel'] = torch.zeros((self.movie_len,3))
        out['orientation'] = torch.zeros((self.movie_len,6))
        out['image'] = torch.zeros((self.movie_len,3,self.cfg['img_size'][1],self.cfg['img_size'][0]))
        out['intrinsics'] = torch.zeros((self.movie_len,3,3))
        out['extrinsics'] = torch.zeros((self.movie_len,4,4))
        out['route_map'] = torch.zeros((self.movie_len,3,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0]))
        out['instance_label'] = torch.zeros((self.movie_len,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0]))
        out['birdview_label'] = torch.zeros((self.movie_len,1,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0])).to(torch.long)
        # out['map_label'] = torch.zeros((self.movie_len,4,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0]))
        for i in range(self.movie_len):
            action = actions[i]
            out['vel'][i] = action[:3]
            out['accel'][i] = action[3:6]
            out['orientation'][i] = action[6:]
            sample_token = video_info[i]['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            log_token = scene['log_token']
            log = self.nusc.get('log',log_token)
            nusc_map = self.nusc_maps[log['location']]
            image,intrinsics = self.get_cam_image_from_sample_token(sample_token,tuple(self.cfg['img_size']))
            out['image'][i],out['intrinsics'][i] = image,intrinsics
            out['extrinsics'][i] = self.get_cam2ego_matrix(sample_token)
            bev_hdmap = get_bev_hdmap_front_view(sample_token,self.nusc,nusc_map,width=76.8,height=76.8,img_size=(self.cfg['hdmap_size'][0]*2,self.cfg['hdmap_size'][1]*2))
            bev_hdmap = bev_hdmap[:,:,:,0].copy()
            bev_hdmap = self.crop_image(bev_hdmap,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0],self.cfg['hdmap_size'][0]//2)
            bev_hdmap = torch.from_numpy(bev_hdmap).to(torch.float32)
            bev_instance_label,bev_box = get_bev_box_label(sample_token,self.nusc,nusc_map,width=76.8,height=76.8,img_size=(self.cfg['hdmap_size'][0]*2,self.cfg['hdmap_size'][1]*2),instance_label=self.instance_label)
            # bev_instance_label = torch.from_numpy(bev_instance_label).to(torch.float32)
            
            bev_box = self.crop_image(bev_box,self.cfg['hdmap_size'][1],self.cfg['hdmap_size'][0],self.cfg['hdmap_size'][0]//2)
            
            instance_mask = bev_box[1].copy().astype(np.bool) | bev_box[2].astype(np.bool)
            instance_label,_ = scipy.ndimage.label(instance_mask.astype(np.int64))
            instance_label = torch.from_numpy(instance_label)
            bev_box = torch.from_numpy(bev_box).to(torch.float32)
            bev_labels = self.calculate_birdview_labels(bev_hdmap,bev_box)
            out['route_map'][i] = bev_hdmap
            # out['map_label'][i] = self.get_map_label(copy.deepcopy(bev_hdmap))
            out['instance_label'][i] = instance_label
            out['birdview_label'][i] = bev_labels
        
        return out
import cv2
def visualize_channels(tensor, save_path):
    """
    将一个 C, H, W 的 tensor 可视化，每个通道用不同的颜色表示。

    Args:
        tensor: 待可视化的 tensor。
        save_path: 保存图片的路径。
    """

    # 将 tensor 转化为 numpy 数组
    img_np = tensor.transpose(1, 2, 0)

    # 定义颜色映射
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    # # 如果通道数大于颜色数量，可以循环使用颜色
    # colors = colors * (img_np.shape[2] // len(colors) + 1)

    # # 为每个通道创建一个图像
    # imgs = []
    # for i in range(img_np.shape[2]):
    #     img = img_np[:, :, i]
    #     img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)  # 应用颜色映射
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    #     imgs.append(img)

    # # 将所有通道的图像水平拼接
    # img_concat = np.hstack(imgs)

    # 保存图像
    cv2.imwrite(save_path, img_np)

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/MILE.yaml',
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

    for _,batch in tqdm(enumerate(data_loader_)):
        pass