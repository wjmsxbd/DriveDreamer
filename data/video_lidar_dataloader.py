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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar,get_this_scene_info_with_lidar_figax
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
import time
from tqdm import tqdm
from einops import rearrange
from memory_profiler import profile
import gc
import copy
import matplotlib.pyplot as plt

try:
    import moxing as mox

    mox.file.shift('os', 'mox')
except:
    pass


class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,movie_len,split_name='train'):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        self.movie_len = movie_len
        self.num_boxes = num_boxes
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot=cfg['dataroot'], map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=cfg['dataroot'], map_name='singapore-queenstown'),
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
    # @profile(precision=4,stream=open('log.txt',"w+",encoding="utf-8"))
    def get_data_info(self,idx):
        video_info = self.video_infos[idx]
        out = {}
        out['reference_image'] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))
        out['HDmap'] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))
        out['range_image'] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))
        out['dense_range_image'] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))
        out['3Dbox'] = torch.zeros((self.movie_len,self.num_boxes,16))
        out['category'] = [None for i in range(self.movie_len)]
        out['text'] = [None for i in range(self.movie_len)]
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
                ref_img,boxes,hdmap,category,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']))
            else:
                ref_img,boxes,hdmap,category,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,fig=self.fig,ax=self.ax)
            dense_range_image = np.repeat(dense_range_image[...,np.newaxis],3,axis=-1)

            boxes = np.array(boxes).astype(np.float32)
            ref_img = torch.from_numpy(ref_img / 255. * 2 - 1.).to(torch.float32)
            hdmap = hdmap[:,:,:3].copy()
            hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.).to(torch.float32)
            range_image = torch.from_numpy(range_image / 255. * 2 - 1.).to(torch.float32)
            dense_range_image = torch.from_numpy(dense_range_image / 255. * 2 - 1.).to(torch.float32)
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
            out['reference_image'][i] = ref_img
            out['HDmap'][i] = hdmap
            out['range_image'][i] = range_image
            out['dense_range_image'][i] = dense_range_image
            out['3Dbox'][i] = boxes
            out['category'][i] = category
            out['text'][i] = text

        return out
    def __getitem__(self,idx):
        return self.get_data_info(idx)

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
                        default='configs/prediction.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)

    data_loader = dataloader(**cfg.data.params.train.params)
    # 35
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size  =   1,
        num_workers =   0,
        collate_fn=collate_fn
    )
    # network = instantiate_from_config(cfg['model'])
    # model_path = 'logs/2024-08-06T07-11-46_global_condition/checkpoints/epoch=000093.ckpt'
    # network.init_from_ckpt(model_path)
    # network = network.eval().cuda()
    # save_path = 'all_pics/'

    # samples = torch.randint(0,len(data_loader),(5,)).long()
    # for sample in samples:
    #     input_data = data_loader.__getitem__(0)
    #     # input_data = {k:v.unsqueeze(0).cuda() for k,v in input_data.items()}
    #     batch = {}
    #     batch['range_image'] = input_data['range_image']
    #     logs = network.log_images(batch)
    #     save_tensor_as_image(logs['inputs'],file_path=save_path+f'inputs_{sample:02d}.jpg')
    #     save_tensor_as_image(logs['samples'],file_path=save_path+f'samples{sample:02d}.jpg')
    #     save_tensor_as_image(logs['reconstructions'],file_path=save_path+f'reconstructions{sample:02d}.jpg')
    # for _ in range(len(data_loader)):
    #     data_loader.__getitem__(_)
    for _,batch in tqdm(enumerate(data_loader_)):
        # batch['reference_image'] = batch['reference_image'][:,0:1]
        # batch['reference_image'] = repeat(batch['reference_image'],'b n h w c -> b (repeat n) h w c',repeat=5)
        # logs = network.log_video(batch)
        # save_tensor_as_video(logs['inputs'],file_path=save_path+f'inputs_{_:02d}.png')
        # save_tensor_as_video(logs['samples'],file_path=save_path+f'samples{_:02d}.png')
        # save_tensor_as_video(logs['lidar_samples'],file_path=save_path+f'lidar_samples{_:02d}.png')
        # save_tensor_as_video(logs['lidar_inputs'],file_path=save_path+f'lidar_inputs{_:02d}.png')
        pass
