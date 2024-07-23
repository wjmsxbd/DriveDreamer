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
from ip_basic.ip_basic.depth_map_utils import fill_in_fast,fill_in_multiscale
from einops import rearrange
import png

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,split_name='train',device=None):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        if device is not None:
            self.device = f'cuda:{device}'
        self.num_boxes = num_boxes
        #torch.cuda.set_device(cfg['device'])

        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        #$self.instantiate_cond_stage(cond_stage_config)
        # self.clip = self.clip.to(self.device)
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
                capture_frequency = (capture_frequency + 1) % self.capture_frequency
        print(len(self.data_infos))
        

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
        if self.cfg['img_size'] is not None:
            ref_img,boxes,hdmap,category,depth_cam_front_img,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']))
        else:
            ref_img,boxes,hdmap,category,depth_cam_front_img,range_image,dense_range_image = get_this_scene_info_with_lidar(self.cfg['dataroot'],self.nusc,nusc_map,sample_token)
        out = {}
        boxes = np.array(boxes).astype(np.float32)
        ref_img = torch.from_numpy(ref_img / 255. * 2 - 1.0).to(torch.float32)
        hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.0).to(torch.float32)

        depth_cam_front_img = torch.from_numpy(depth_cam_front_img / 255. * 2 - 1.0).to(torch.float32)
        range_image = torch.from_numpy(range_image / 255. * 2 - 1.0).to(torch.float32)
        dense_range_image = repeat(dense_range_image,'h w -> h w c', c=3)
        out['dense_range_image'] = torch.from_numpy(dense_range_image / 255. * 2 - 1.0).unsqueeze(0)
        out['range_image'] = range_image[:,:,:3].unsqueeze(0)
        out['depth_cam_front_img'] = depth_cam_front_img.unsqueeze(0)
        out['reference_image'] = ref_img.unsqueeze(0)
        out['HDmap'] = hdmap[:,:,:3].unsqueeze(0)
        out['text'] = [text]
        if boxes.shape[0] == 0:
            boxes = torch.from_numpy(np.zeros((self.num_boxes,16)))
            category = ["None" for i in range(self.num_boxes)]
        elif boxes.shape[0]<self.num_boxes:
            zero_len = self.num_boxes - boxes.shape[0]
            boxes_zero = np.zeros((zero_len,16))
            boxes = torch.from_numpy(np.concatenate((boxes,boxes_zero),axis=0))
            category_none = ["None" for i in range(zero_len)]
            category = category + category_none
        else:
            boxes = torch.from_numpy(boxes[:self.num_boxes])
            category_embed = category[:self.num_boxes]
            category = category_embed[:self.num_boxes]
        out['3Dbox'] = boxes.unsqueeze(0).to(torch.float32)
        out['category'] = category
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
                        default='configs/global_condition.yaml',
                        type=str,
                        help="config path")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    cfg.data.params.train.params['device'] = 0
    data_loader = dataloader(**cfg.data.params.train.params)
    # 35
    out = data_loader.__getitem__(30)
    range_image = out['dense_range_image']
    file_path = "001.png"
    # image = Image.fromarray(range_image)

    # image.save(file_path)
    # with open(file_path, 'wb') as f:
    #     range_image = (range_image*256).astype(np.uint16)
    #     writer = png.Writer(width=range_image.shape[1],
    #                         height=range_image.shape[0],
    #                         bitdepth=16,
    #                         greyscale=True)
    #     writer.write(f, range_image)

    # range_image = rearrange(range_image,'b h w c -> b c h w')
    # depth_cam_front_img = rearrange(out['depth_cam_front_img'],'b h w c -> b c h w')
    # range_image = rearrange(out['range_image'],'b h w c -> b c h w')
    # dense_range_image = rearrange(out['dense_range_image'],'b h w c -> b c h w')
    # reference_image = rearrange(out['reference_image'],'b h w c -> b c h w')
    # HDmap = rearrange(out['HDmap'],'b h w c -> b c h w')
    # save_tensor_as_image(range_image,file_path="./temp/003.png")
    # save_tensor_as_image(depth_cam_front_img,file_path="./temp/depth_cam_front_img.png")
    # save_tensor_as_image(range_image,file_path="./temp/range_image.png")
    # save_tensor_as_image(dense_range_image,file_path="./temp/dense_range_image.png")
    # save_tensor_as_image(reference_image,file_path="./temp/reference_image.png")
    # save_tensor_as_image(HDmap,file_path="./temp/HDmap.png")
    
    # range_image = range_image.numpy()[0,0,:,:]
    # range_image,_ = fill_in_multiscale(range_image)
    # range_image = torch.from_numpy(range_image)
    # range_image = range_image.unsqueeze(0).repeat(3,1,1)
    # save_tensor_as_image(range_image,file_path="001.jpg")
    network = instantiate_from_config(cfg['model']['params']['global_condition_config']['params']['lidar_config'])
    model_path = 'stable_diffusion/kl-f8.ckpt'
    network.init_from_ckpt(model_path)
    network = network.eval().cuda()
    save_path = 'all_pics/'
    samples = torch.randint(0,len(data_loader),(5,)).long()
    for sample in samples:
        input_data = data_loader.__getitem__(0)
        # input_data = {k:v.unsqueeze(0).cuda() for k,v in input_data.items()}
        batch = {}
        batch['range_image'] = input_data['range_image']
        logs = network.log_images(batch)
        save_tensor_as_image(logs['inputs'],file_path=save_path+f'inputs_{sample:02d}.jpg')
        save_tensor_as_image(logs['samples'],file_path=save_path+f'samples{sample:02d}.jpg')
        save_tensor_as_image(logs['reconstructions'],file_path=save_path+f'reconstructions{sample:02d}.jpg')
        

    
