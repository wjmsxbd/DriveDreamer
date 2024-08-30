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
from tqdm import tqdm
from einops import rearrange
# from memory_profiler import profile
# import tracemalloc

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
                ref_img,boxes,hdmap,category,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']))
            else:
                ref_img,boxes,hdmap,category,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token)
            
            ref_img = torch.from_numpy(ref_img / 255. * 2 - 1.).to(torch.float32)
            

            if not 'reference_image' in out.keys():
                out['reference_image'] = ref_img.unsqueeze(0)
            else:
                out['reference_image'] = torch.cat([out['reference_image'],ref_img.unsqueeze(0)],dim=0)

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
                        default='configs/video_decoder.yaml',
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
    # model_path = 'stable_diffusion/vista.safetensors'
    # ckpt_path = 'stable_diffusion/kl-f8.ckpt'
    # network.init_from_ckpt(ckpt_path)
    # network.init_from_safetensor(model_path)
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
    import objgraph
    for id,batch in tqdm(enumerate(data_loader_)):
        # objgraph.show_backrefs(objgraph.by_type('SplitResult')[0], max_depth = 10, filename = 'obj.dot')
        # objgraph.show_growth()
        # tmp = batch['reference_image'].clone()
        # b = tmp.shape[0]
        # tmp = rearrange(tmp,'b n h w c -> b n c h w')
        # input = network.get_input(batch,'reference_image').cuda()
        # dec,_ = network(input)
        # save_tensor_as_video(tmp,file_path=save_path+'inputs_{}.png'.format(id))
        # dec = dec.reshape((b,-1)+dec.shape[1:])
        # save_tensor_as_video(dec,file_path=save_path+'samples_{}.png'.format(id))
        # objgraph.show_growth()
        pass