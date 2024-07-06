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
from utils.tools import get_this_scene_info_with_lidar
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
from utils.tools import render_pointcloud_in_image,convert_fig_to_numpy

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,split_name='train',device=None):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        if device is not None:
            self.device = f'cuda:{device}'
        # self.num_boxes = num_boxes
        # self.nusc_maps = {
        #     'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
        #     'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
        #     'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
        #     'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        # }
        #$self.instantiate_cond_stage(cond_stage_config)
        self.capture_frequency = cfg.capture_frequency
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        with open(data_info_path,'rb') as f:
            data_infos = pickle.load(f)
        sorted_data_infos = list(sorted(data_infos['infos'],key=lambda e:e['timestamp']))
        scene_infos = dict()
        for data_info in sorted_data_infos:
            sample_token = data_info['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            if not scene['name'] in scene_infos.keys():
                scene_infos[scene['name']] = [data_info]
            else:
                scene_infos[scene['name']].append(data_info)
        
        self.data_infos = []
        capture_frequency = 0
        for scene_name,data_infos in scene_infos.items():

            for data_info in data_infos:
                if capture_frequency == 0:
                    self.data_infos.append(data_info)
                    self.data_infos.append(data_info)
                capture_frequency = (capture_frequency + 1) % self.capture_frequency
        
        

    def __len__(self):
        return len(self.data_infos)


    def get_data_info(self,idx):
        sample_token = self.data_infos[idx]['token']
        scene_token = self.nusc.get('sample',sample_token)['scene_token']
        scene = self.nusc.get('scene',scene_token)
        out = {}
        #TODO: scale [-1,1] and batch size
        if idx % 2 == 0:
            range_image_fig,range_image_ax = render_pointcloud_in_image(self.nusc,sample_token)
            range_image = convert_fig_to_numpy(range_image_fig,(1600,900))
            range_image = Image.fromarray(range_image)
            if self.cfg['img_size'] is not None:
                range_image = np.array(range_image.resize(self.cfg['img_size']))
            else:
                range_image = np.array(range_image)
            out['reference_image'] = range_image[:,:,:3] / 255. * 2 - 1.0
        else:
            sample_record = self.nusc.get('sample',sample_token)
            cam_front_token = sample_record['data']['CAM_FRONT']
            cam_front_path = self.nusc.get('sample_data',cam_front_token)['filename']
            depth_cam_front_path = cam_front_path.split('/')
            depth_cam_front_path[0] = depth_cam_front_path[0] + '_depth'
            depth_cam_front_path[-1] = depth_cam_front_path[-1].split('.')[0] + '_depth.png'
            depth_cam_front_path = os.path.join(self.cfg['dataroot'],*depth_cam_front_path)
            depth_cam_front_img = mpimg.imread(depth_cam_front_path).astype(np.float32)
            depth_cam_front_img = (depth_cam_front_img * 255).astype(np.uint8)
            depth_cam_front_img = Image.fromarray(depth_cam_front_img)
            if self.cfg['img_size'] is not None:
                depth_cam_front_img = np.array(depth_cam_front_img.resize(self.cfg['img_size']))
            else:
                depth_cam_front_img = np.array(depth_cam_front_img)
            out['reference_image'] = depth_cam_front_img / 255. * 2 - 1.0
        return out

    def __getitem__(self,idx):
        return self.get_data_info(idx)
    
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


if __name__ == '__main__':
    # _____________________________________________________________
    # import argparse
    # parser = argparse.ArgumentParser(description='AutoDM-training')
    # parser.add_argument('--config',
    #                     default='configs/first_stage_step1_config_online2.yaml',
    #                     type=str,
    #                     help="config path")
    # cmd_args = parser.parse_args()
    # cfg = omegaconf.OmegaConf.load(cmd_args.config)
    # # print("get cfg!!!!!!!!!!!!")
    # cfg.data.params.train.params['device'] = 0
    # data_loader = dataloader(**cfg.data.params.train.params)
    # network = instantiate_from_config(cfg['model'])
    # model_path = 'logs/2024-05-24T17-05-15_first_stage_step1_config_online2/checkpoints/epoch=000008.ckpt'
    # # model_path = 'logs/2024-05-25T05-27-43_first_stage_step1_config_online3/checkpoints/epoch=000000.ckpt'
    # network.init_from_ckpt(model_path)
    # network = network.eval().cuda()
    # save_path = 'myencodersd/'
    # # save_path = 'sd_images/'
    # for i in range(200,250,10):
    #     input_data = data_loader.__getitem__(i)
    #     input_data = {k:v.unsqueeze(0).cuda() for k,v in input_data.items()}
    #     logs = network.log_images(input_data)
    #     save_tensor_as_image(logs['inputs'],file_path=save_path+f'inputs_{i:02d}.jpg')
    #     save_tensor_as_image(logs['samples'],file_path=save_path+f'samples{i:02d}.jpg')
    #     # logs['hdmap'] = logs['hdmap'][:,:3]
    #     # print(logs['hdmap'].shape)
    #     # save_tensor_as_image(logs['hdmap'],file_path=save_path+f'hdmap{i:02d}.jpg')
    # # out = data_loader.__getitem__(0)
    # # print([v.shape for k,v in out.items()])
    # # print(out.keys())
    
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/first_stage_step1_config_high.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    cfg.data.params.train.params['device'] = 0
    data_loader = dataloader(**cfg.data.params.train.params)
    out1 = data_loader.__getitem__(0)
    out2 = data_loader.__getitem__(1)
    ref_img1 = out1['reference_image']
    ref_img2 = out2['reference_image']
    ref_img1 = Image.fromarray(ref_img1)
    ref_img2 = Image.fromarray(ref_img2)
    ref_img1.save("ref_img1.png")
    ref_img2.save("ref_img2.png")
    # network = instantiate_from_config(cfg['model'])
    # model_path = 'logs/2024-06-24T07-06-16_first_stage_step1_config_mini/checkpoints/last.ckpt'
    # network.init_from_ckpt(model_path)
    # network = network.eval().cuda()
    # save_path = 'all_pics/add_night_condition/'
    # for i in range(1,2001,10000):
    #     input_data = data_loader.__getitem__(i)
    #     input_data = {k:v.unsqueeze(0).cuda() for k,v in input_data.items()}
    #     logs = network.log_images(input_data)
    #     save_tensor_as_image(logs['inputs'],file_path=save_path+f'inputs_{i:02d}.jpg')
    #     save_tensor_as_image(logs['samples'],file_path=save_path+f'samples{i:02d}.jpg')
        

    
