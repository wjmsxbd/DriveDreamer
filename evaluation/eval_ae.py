import sys

import torch.utils.data.distributed
sys.path.append('.')
sys.path.append('.')
import torch
import torch.distributed as dist
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
from utils.tools import get_this_scene_info,get_this_scene_info_with_lidar,get_gt_lidar_point
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
import imageio
from einops import rearrange,repeat
from tqdm import tqdm
import copy
from pytorch_fid import fid_score
import scipy.spatial as spt
import glob
# try:
#     from evaluation.lidar_bonnetal.train.tasks.semantic.FRID import FRIDCalculation
# except:
#     pass
from evaluation.lidar_bonnetal.train.tasks.semantic.FRID import FRIDCalculation

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,movie_len,split_name='train',return_pose_info=False,collect_condition=None,use_original_action=False):
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
        batch = self.get_data_info(idx)
        batch['index'] = idx
        return batch
    
    def get_gt_lidar_point(self,sample_token,return_intensity=False):
        gt_points = []
        for bs in range(len(sample_token)):
            points = []
            for token in sample_token[bs]:
                points.append(get_gt_lidar_point(self.nusc,token,return_intensity=return_intensity))
            gt_points.append(points)
        return gt_points

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
                if self.use_original_action:
                    out[key] = torch.zeros((self.movie_len,7))
                else:    
                    out[key] = torch.zeros((self.movie_len,14))
            # elif key == 'bev_images':
            #     out[key] = torch.zeros((self.movie_len,self.cfg['bev_size'][0],self.cfg['bev_size'][1],3))
            else:
                out[key] = torch.zeros((self.movie_len,self.cfg['img_size'][1],self.cfg['img_size'][0],3))
        out['sample_token'] = [None for i in range(self.movie_len)]
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
            
            out['sample_token'][i] = sample_token
            if self.use_original_action:
                now_action = actions[i]
            else:
                cam_front_translation = torch.tensor(collect_data['cam_front_record']['translation'])
                cam_front_rotation = torch.tensor(collect_data['cam_front_record']['rotation'])
                cam_front_intrinsic = torch.tensor([collect_data['cam_front_record']['camera_intrinsic']],dtype=torch.float32).flatten()
                # now_action = torch.cat([actions[i],cam_front_translation,cam_front_rotation,cam_front_intrinsic],dim=-1).unsqueeze(0)
                now_action = torch.cat([actions[i],cam_front_translation,cam_front_rotation],dim=-1).unsqueeze(0)
            

            if 'text' in self.collect_condition:
                out['text'][i] = text
            for key in collect_data:
                if key == '3Dbox':
                    boxes = collect_data['3Dbox']
                    category = collect_data['category']
                    boxes = np.array(boxes).astype(np.float32)
                    # add_box = np.array([0.25,0.23,0.23,0.25,0.25,0.23,0.23,0.25,0.52,0.52,0.62,0.62,0.52,0.52,0.62,0.62]).astype(np.float32)
                    # add_category = 'human.pedestrian.adult'
                    # if boxes.shape[0] >= self.num_boxes:
                    #     boxes[self.num_boxes-1] = add_box
                    #     category[self.num_boxes-1] = add_category
                    # else:
                    #     boxes = np.concatenate([boxes,add_box[np.newaxis,:]],axis=0)
                    #     category.append(add_category)
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
            elif isinstance(value,(int,float)):
                if not key in out.keys():
                    out[key] = []
                    out[key].append(value)
                else:
                    out[key].append(value)
            else:
                raise NotImplementedError
    return out

def save_tensor_as_image(tensor, file_path,index):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.clamp(-1, 1)  # 确保值在[0, 1]之间
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor * 255.0
    tensor = tensor.byte()

    if len(tensor.shape) == 5:
        for idx in range(tensor.shape[0]):
            for frame in range(tensor.shape[1]):
                save_file_path = os.path.join(file_path,f'{index[idx]:02d}_{frame:02d}.png')
                img = tensor[idx,frame]
                img = img.permute(1,2,0)
                img = img.numpy()
                img = Image.fromarray(img)
                img.save(save_file_path)                
    
def save_tensor_as_video(tensor,file_path,index):
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
        save_video_path = os.path.join(file_path,f'{index[idx]:02d}.gif')
        video = [tensor[i,j] for j in range(tensor.shape[1])]
        imageio.mimsave(save_video_path,video,'GIF',fps=5,loop=0)

def evaluate_fid(real_save_path,rec_save_path):
    fid_value = fid_score.calculate_fid_given_paths([real_save_path,rec_save_path],2,'cuda',2048)
    return fid_value

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1,1,3,1,1)
    x = x.permute(0,2,1,3,4)
    return x

def evaluate_fvd(real_save_path,rec_save_path):
    with torch.no_grad():
        from FVD.fvdcal import FVDCalculation
        from pathlib import Path
        fvd_videogpt = FVDCalculation(method='videogpt')
        # fvd_stylegan = FVDCalculation(method='stylegan')
        generated_videos_folder = Path(rec_save_path)
        real_videos_folder = Path(real_save_path)

        videos_list1 = list(real_videos_folder.glob("*.gif"))
        videos_list2 = list(generated_videos_folder.glob("*.gif"))
        score_videogpt = fvd_videogpt.calculate_fvd_by_video_path(videos_list1, videos_list2)
        print(score_videogpt)
        # score_stylegan = fvd_stylegan.calculate_fvd_by_video_path(videos_list1, videos_list2)
        # print(score_stylegan)

def evaluate_psnr(real_save_path,rec_save_path):
    real_pic_path = glob.glob(os.path.join(real_save_path,'*.png'))
    rec_pics_path = glob.glob(os.path.join(rec_save_path,'*.png'))
    real_pics = []
    for pic_path in real_pic_path:
        img = mpimg.imread(pic_path)
        real_pics.append(img)
    real_pics = np.stack(real_pics)
    rec_pics = []
    for pic_path in rec_pics_path:
        img = mpimg.imread(pic_path)
        rec_pics.append(img)
    rec_pics = np.stack(rec_pics)
    mse = np.mean((real_pics - rec_pics) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(real_pics,axis=(1,2,3))
    psnr = 10 * np.log10((max_pixel)**2 / mse)
    return np.mean(psnr)

def evaluate_ssim(real_save_path,rec_save_path,window_size=11,size_average=True,full=False):
    real_pic_path = glob.glob(os.path.join(real_save_path,'*.png'))
    rec_pics_path = glob.glob(os.path.join(rec_save_path,'*.png'))
    real_pics = []
    for pic_path in real_pic_path:
        img = mpimg.imread(pic_path)
        img = img.transpose(2,0,1)
        real_pics.append(img)
    real_pics = np.stack(real_pics)
    real_pics = torch.from_numpy(real_pics)
    rec_pics = []
    for pic_path in rec_pics_path:
        img = mpimg.imread(pic_path)
        img = img.transpose(2,0,1)
        rec_pics.append(img)
    rec_pics = np.stack(rec_pics)
    rec_pics = torch.from_numpy(rec_pics)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    window = torch.ones(3,1,window_size,window_size).float()
    mu1 = torch.nn.functional.conv2d(real_pics,window,padding=window_size//2,groups=3)
    mu2 = torch.nn.functional.conv2d(rec_pics,window,padding=window_size//2,groups=3)

    mu1_sq,mu2_sq,mu1_mu2 = mu1 ** 2,mu2 ** 2,mu1*mu2
    sigma1_sq = torch.nn.functional.conv2d(real_pics**2,window,padding=window_size//2,groups=3) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(rec_pics**2,window,padding=window_size//2,groups=3) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(real_pics*rec_pics,window,padding=window_size//2,groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        ssim_value = torch.mean(ssim_map)
    else:
        ssim_value = torch.mean(ssim_map,dim=(1,2,3))
    if full:
        return ssim_value,ssim_map
    else:
        return ssim_value

import copy
def calc_point_cloud_distance(gt_lidar_point,inputs,reconstructions,bs=1,movie_len=5):
    inputs = inputs.cpu()
    reconstructions = reconstructions.cpu()
    img_size = [256,128]
    reconstructions = (torch.clip(reconstructions,-1,1) + 1.) / 2.
    mask = reconstructions >= 0.05
    reconstructions = reconstructions * mask
    reconstructions = (reconstructions * 255.)

    inputs = (torch.clip(inputs,-1,1) + 1.) / 2.
    inputs = (inputs * 255.)
    copy_inputs = copy.deepcopy(inputs[:,:,0])
    copy_rec = copy.deepcopy(reconstructions[:,:,2])

    theta_up = torch.pi / 12
    theta_down = -torch.pi / 6
    theta_res = (theta_up - theta_down) / img_size[0] / 2
    phi_res = (torch.pi / 3) / img_size[1] / 2
    b,n,c,h,w = inputs.shape
    x_coords,y_coords = torch.meshgrid(torch.arange(0,h),torch.arange(0,w))
    inputs[:,:,0] = x_coords
    inputs[:,:,1] = y_coords

    reconstructions[:,:,0] = x_coords
    reconstructions[:,:,1] = y_coords

    inputs_phi = ((1+inputs[:,:,0:1]) / img_size[1] - 1) * torch.pi / 6 + phi_res
    reconstructions_phi = ((1+reconstructions[:,:,0:1]) / img_size[1] - 1) * torch.pi / 6 + + phi_res

    inputs_theta = (theta_up) - (theta_up - theta_down) * ((inputs[:,:,1:2] + 1) - 1./2) / img_size[0]  + theta_res
    reconstructions_theta = (theta_up) - (theta_up - theta_down) * ((reconstructions[:,:,1:2] + 1) - 1./2) / img_size[0] + theta_res

    inputs_r = inputs[:,:,2:3]
    reconstructions_r = reconstructions[:,:,2:3]

    point_x_inputs = inputs_r * torch.cos(inputs_theta) * torch.sin(inputs_phi)
    point_y_inputs = inputs_r * torch.cos(inputs_theta) * torch.cos(inputs_phi)
    point_z_inputs = inputs_r * torch.sin(inputs_theta)
    points_inputs = torch.cat([point_x_inputs,point_y_inputs,point_z_inputs],dim=2)

    point_x_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.sin(reconstructions_phi)
    point_y_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.cos(reconstructions_phi)
    point_z_rec = reconstructions_r * torch.sin(reconstructions_theta)
    points_rec = torch.cat([point_x_rec,point_y_rec,point_z_rec],dim=2)

    # points_inputs = rearrange(points_inputs,'b n c h w -> b n (h w) c')
    input_point_loss = torch.tensor([0]).to(torch.float32)
    rec_point_loss = torch.tensor([0]).to(torch.float32)
    for batch_id in range(bs):
        for frame in range(movie_len):
            input_indices = torch.nonzero(copy_inputs[batch_id][frame])
            rec_indices = torch.nonzero(copy_rec[batch_id][frame])
            kd_tree = spt.cKDTree(data=gt_lidar_point[batch_id][frame][:,:3])
            for idx in range(len(input_indices)):
                x,y = input_indices[idx]
                point = points_inputs[batch_id][frame][:,x,y]
                dis,point_id = kd_tree.query(point)
                input_point_loss = input_point_loss + dis
            input_point_loss = input_point_loss / len(input_indices)
            for idx in range(len(rec_indices)):
                x,y = rec_indices[idx]
                point = points_rec[batch_id][frame][:,x,y]
                dis,point_id = kd_tree.query(point)
                rec_point_loss += dis
            rec_point_loss /= len(rec_indices)
        input_point_loss /= movie_len
        rec_point_loss /= movie_len
    input_point_loss /= bs
    rec_point_loss /= bs
    print(f"calc input point loss:{input_point_loss}")
    print(f"calc rec point loss:{rec_point_loss}")
    return rec_point_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--config',
                        default='configs/svd_2gpus.yaml',
                        type=str,
                        help="config path")
    parser.add_argument('--video',
                        action='store_true',
                        help="use video evaluation")
    parser.add_argument('--train',
                        action='store_true',
                        help="use video evaluation")
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help="device")
    parser.add_argument('--cuda_id',
                        default='0,',
                        type=str,
                        help="cuda_id")
    parser.add_argument('--type',
                        default='baseline',
                        type=str,
                        help="save_path")
    parser.add_argument('--model_path',
                        default=None,
                        type=str,
                        help="model_path")
    parser.add_argument('--only_camera',
                        action='store_true',
                        help="only_camera")
    parser.add_argument('--n_samples',
                        default=1,
                        type=int,
                        help="video sample step")
    parser.add_argument('--local-rank',
                        default=1,
                        type=int,
                        help="local_rank")
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)
    torch.manual_seed(23)
    video_eval = cmd_args.video
    device = cmd_args.device
    use_train = cmd_args.train
    path_type = cmd_args.type
    only_camera = cmd_args.only_camera
    cuda_id = cmd_args.cuda_id.split(',')
    local_rank = cmd_args.local_rank
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group('nccl',world_size=world_size,rank=rank)

    if use_train:
        data_loader = dataloader(**cfg.data.params.train.params)
    else:
        data_loader = dataloader(**cfg.data.params.validation.params)
    # 35
    sampler = torch.utils.data.distributed.DistributedSampler(data_loader)
    batch_size = 2
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size  =   batch_size,
        num_workers =   0,
        sampler=sampler,
        collate_fn=collate_fn
    )
    network = instantiate_from_config(cfg['model'])
    model_path = cmd_args.model_path
    if model_path:
        network.init_from_ckpt(model_path)
    if device == 'cuda':
        print(f"now_process:{local_rank}")
        network = network.eval().to(f'cuda:{cuda_id[local_rank]}')
    save_path = '/storage/group/4dvlab/wangjm2024/all_pics/'
    save_path = os.path.join(save_path,path_type)

    cam_real_save_path = save_path + '/cam_inputs/'
    cam_rec_save_path = save_path + "/cam_rec/"
    cam_samples_save_path = save_path + "/cam_samples/"
    lidar_real_save_path = save_path + '/lidar_inputs/'
    lidar_rec_save_path = save_path + "/lidar_rec/"
    lidar_samples_save_path = save_path + "/lidar_samples/"
    if rank == 0:
        if not os.path.exists(cam_real_save_path):
            os.makedirs(cam_real_save_path)
        if not os.path.exists(cam_rec_save_path):
            os.makedirs(cam_rec_save_path)
        if not os.path.exists(lidar_real_save_path):
            os.makedirs(lidar_real_save_path)
        if not os.path.exists(lidar_rec_save_path):
            os.makedirs(lidar_rec_save_path)
    dist.barrier()
    videos1 = []
    videos2 = []
    # if not video_eval:
    #     cfg_path = 'evaluation/lidar_bonnetal/train/tasks/semantic/log2/arch_cfg.yaml'
    #     model_path = 'evaluation/lidar_bonnetal/train/tasks/semantic/log2'
    #     data_cfg_path = 'evaluation/lidar_bonnetal/train/tasks/semantic/log2/data_cfg.yaml'
    #     frid = FRIDCalculation(cfg_path,data_cfg_path,model_path=model_path,device=(f'cuda:{cuda_id}' if device == 'cuda' else 'cpu'))
    #     real_features = []
    #     rec_features = []
    point_losses = []

    for _,batch in tqdm(enumerate(data_loader_)):
        idx = batch['index']
        del batch['index']
        if device == 'cuda':
            batch = {k:v.to(f'cuda:{cuda_id[local_rank]}') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        if not video_eval and not only_camera:
            gt_lidar_point = data_loader.get_gt_lidar_point(batch['sample_token'],return_intensity=False)
        del batch['sample_token']
        logs = network.log_video(batch)
        # if not video_eval and not only_camera:
        #     real_feature,rec_feature = frid.calculate_frid_by_video_path(logs['lidar_inputs'],logs['lidar_reconstruction'],gt_lidar_point)
        #     real_features.append(real_feature)
        #     rec_features.append(rec_feature)
        # depth = torch.mean(logs['lidar_reconstruction'],dim=2)
        # logs['lidar_reconstruction'][:,:,:3] = depth
        if not video_eval:
            save_tensor_as_image(logs['inputs'],file_path=cam_real_save_path,index=idx)
            save_tensor_as_image(logs['samples'],file_path=cam_rec_save_path,index=idx)
            if not only_camera:
                save_tensor_as_image(logs['lidar_samples'],file_path=lidar_rec_save_path,index=idx)
                save_tensor_as_image(logs['lidar_inputs'],file_path=lidar_real_save_path,index=idx)
        else:
            if logs['inputs'].shape[1] <10:
                for key in logs.keys():
                    temp = logs[key][:,:1]
                    temp = repeat(temp,'b n c h w -> b (repeat n) c h w',repeat=10-logs['inputs'].shape[1])
                    logs[key] = torch.cat([temp,logs[key]],dim=1)
            save_tensor_as_video(logs['inputs'],file_path=cam_real_save_path,index=idx)
            save_tensor_as_video(logs['samples'],file_path=cam_rec_save_path,index=idx)
            if not only_camera:
                save_tensor_as_video(logs['lidar_samples'],file_path=lidar_rec_save_path,index=idx)
                save_tensor_as_video(logs['lidar_inputs'],file_path=lidar_real_save_path,index=idx)
        # # logs['inputs'] = torch.cat([repeat(logs["inputs"][:,0:1],'b n c h w -> b (repeat n) c h w',repeat=5),logs['inputs']],dim=1)
        # # logs['reconstruction'] = torch.cat([repeat(logs["reconstruction"][:,0:1],'b n c h w -> b (repeat n) c h w',repeat=5),logs['reconstruction']],dim=1)
        # # results_cam = evaluate_fvd(logs['inputs'],logs['reconstruction'],'cuda')
        # # print(results_cam)
        # # results_lidar = evaluate_fvd(logs['lidar_inputs'],logs['lidar_reconstruction'],device)
        # # print(results_lidar)
        # # batch['reference_image'] = batch['reference_image'][:,0:1]
        # # batch['reference_image'] = repeat(batch['reference_image'],'b n h w c -> b (repeat n) h w c',repeat=5)
        # # logs = network.log_video(batch)
        # lidar_inputs = logs['lidar_inputs']
        # lidar_inputs = rearrange(lidar_inputs,'b n c h w -> (b n) c h w')
        # lidar_reconstruction = logs['lidar_reconstruction']
        # lidar_reconstruction = rearrange(lidar_reconstruction,'b n c h w -> (b n) c h w')
        # save_tensor_as_video(logs['inputs'],file_path=cam_real_save_path+f'inputs_{_:02d}.png',save_type='png')
        # save_tensor_as_video(logs['reconstruction'],file_path=cam_rec_save_path+f'samples{_:02d}.png',save_type='png')
        # save_tensor_as_video(logs['lidar_reconstruction'],file_path=lidar_rec_save_path+f'lidar_samples{_:02d}.png',save_type='png')
        # save_tensor_as_video(logs['lidar_inputs'],file_path=lidar_real_save_path+f'lidar_inputs{_:02d}.png',save_type='png')
        # # network.lidar_model.calc_3D_point_loss(lidar_inputs,lidar_reconstruction)
        # # save_tensor_as_video(logs['inputs'],file_path=cam_real_save_path+f'inputs_{_:02d}.gif')
        # # save_tensor_as_video(logs['reconstruction'],file_path=cam_rec_save_path+f'samples{_:02d}.gif')
        # # save_tensor_as_video(logs['lidar_reconstruction'],file_path=lidar_rec_save_path+f'lidar_samples{_:02d}.gif')
        # # save_tensor_as_video(logs['lidar_inputs'],file_path=lidar_real_save_path+f'lidar_inputs{_:02d}.gif')

        if not video_eval and not only_camera:
            point_loss = calc_point_cloud_distance(gt_lidar_point,logs['lidar_inputs'],logs['lidar_reconstruction'],movie_len=data_loader.movie_len).numpy()
            point_losses.append(point_loss)

        # if _>1:
        #     break
    dist.barrier()
    if rank==0:
        if not video_eval and not only_camera:
            point_losses = np.vstack(point_losses)
            print(f"point loss mean:{np.mean(point_losses)}")
        if not video_eval:
            # if not only_camera:
            #     real_features = np.vstack(real_features)
            #     rec_features = np.vstack(rec_features)
            #     real_mu = np.mean(real_features,axis=0)
            #     real_sigma = np.cov(real_features,rowvar=False)
            #     rec_mu = np.mean(rec_features,axis=0)
            #     rec_sigma = np.cov(rec_features,rowvar=False)
            #     fid_value = frid.calculate_frechet_distance(real_mu,real_sigma,rec_mu,rec_sigma)
            #     print(f"lidar_frid:{fid_value}")
            cam_fid = evaluate_fid(cam_real_save_path,cam_rec_save_path)
            if not only_camera:
                lidar_fid = evaluate_fid(lidar_real_save_path,lidar_rec_save_path)
                lidar_psnr = evaluate_psnr(cam_real_save_path,cam_rec_save_path)
                lidar_ssim = evaluate_ssim(lidar_real_save_path,lidar_rec_save_path)
            else:
                lidar_fid = 0
                lidar_psnr = 0
                lidar_ssim = 0
            print(f"cam_fid:{cam_fid} lidar_fid:{lidar_fid}")
            cam_psnr = evaluate_psnr(cam_real_save_path,cam_rec_save_path)
            print(f"cam_psnr:{cam_psnr},lidar_psnr:{lidar_psnr}")
            cam_ssim = evaluate_ssim(cam_real_save_path,cam_rec_save_path)
            print(f"cam_ssim:{cam_ssim},lidar_ssim:{lidar_ssim}")

        else:
            cam_fvd = evaluate_fvd(cam_real_save_path,cam_rec_save_path)
            if not only_camera:
                lidar_fvd = evaluate_fvd(lidar_real_save_path,lidar_rec_save_path)
            else:
                lidar_fvd = 0
            print(f"cam_fvd:{cam_fvd},lidar_fvd:{lidar_fvd}")