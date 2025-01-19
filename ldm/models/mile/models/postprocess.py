import scipy.ndimage
import torch
import torch.nn as nn
import numpy as np
import scipy
from pyquaternion import Quaternion
from utils.tools import Point,rotation_6d_to_quaternion,rotation_6d_to_matrix,draw_box_in_camera_view
from nuscenes.utils.geometry_utils import view_points,BoxVisibility
from scipy.spatial import KDTree
import cv2
from PIL import Image

class PostProcess(nn.Module):
    def __init__(self,cfg):
        super(PostProcess,self).__init__()
        self.cfg = cfg
        self.rgb_size = cfg['PostProcess']['RGB']['SIZE']
        self.condition_size = cfg['PostProcess']['CONDITION']['SIZE']
        self.bev_size = cfg['PostProcess']['BEV']['SIZE']
        self.bev_resolution = cfg['PostProcess']['BEV']['RESOLUTION']
        self.num_boxes = cfg['PostProcess']['BOX']['SIZE']
        self.speed_normalization = cfg['SPEED']['NORMALIZATION']
        self.num_cameras = cfg['BEV']['N_CAMERAS']
        
    def check_out_windows(self,point:Point,imsize):
        return point.x < 0 or point.x >=imsize[0] or point.y < 0 or point.y >= imsize[1]
    
    def box_in_image(self,corners,intrinsic,imsize,vis_level:int=BoxVisibility.ANY) -> bool:
        corners_img = view_points(corners,intrinsic,normalize=True)[:2,:]
        visible = np.logical_and(corners_img[0,:] > 0,corners_img[0,:] < imsize[0])
        visible = np.logical_and(visible,corners_img[1,:] < imsize[1])
        visible = np.logical_and(visible,corners_img[1,:] > 0)
        visible = np.logical_and(visible,corners[2,:] > 1)
        in_front = corners[2,:] > 0.1
        if vis_level == BoxVisibility.ALL:
            return all(visible) and all(in_front)
        elif vis_level == BoxVisibility.ANY:
            return any(visible) and all(in_front)
        elif vis_level == BoxVisibility.NONE:
            return True
        else:
            raise ValueError("vis_level: {} not valid".format(vis_level))
        
    def get_box_in_image(self,corners,camera_intrinsics,imsize):
        if not self.box_in_image(corners,camera_intrinsics,imsize):
            return
        corners = view_points(corners,camera_intrinsics,normalize=True)
        corners = corners[:2]
        corners[0] = corners[0] / imsize[0]
        corners[1] = corners[1] / imsize[1]
        return corners
    
    def get_hdmap_in_image(self,image,points,camera_intrinsics,imsize,neighbors,color):
        near_plane = 1e-8
        depths = points[2,:]
        behind = depths < near_plane
        if np.all(behind):
            return
        if len(points) == 0 or points.shape[1] < 2:
            return
        points = view_points(points,camera_intrinsics,normalize=True)
        inside = np.ones(points.shape[1],dtype=bool)
        inside = np.logical_and(inside,points[0,:] > 1)
        inside = np.logical_and(inside,points[0,:] < imsize[0] - 1)
        inside = np.logical_and(inside,points[1,:]>1)
        inside = np.logical_and(inside,points[1,:]<imsize[1]-1)
        points = points[:2].transpose(1,0).astype(np.int16)
        for i in range(points.shape[0]):
            if not inside[i]:
                continue
            for idx in neighbors[i]:
                cv2.line(image,points[i],points[idx],color=color,thickness=1)


    def get_pred_box(self,image,translation,rotation,intrinsics,global2ego_rotation):
        b,n,h,w = image.shape
        n_cam = self.num_cameras
        image = image.view(b*n,h,w)
        x,y = np.meshgrid(
            np.arange(h).astype(np.int16),
            np.arange(w).astype(np.int16)
        )
        translation = translation.view(b*n*n_cam,4,1)[:,:3].cpu().numpy()
        rotation = rotation.view(b*n*n_cam,3,3).cpu().numpy()
        intrinsics = intrinsics.view(b*n*n_cam,3,3).cpu().numpy()
        global2ego_rotation = global2ego_rotation.view(b*n,4).cpu().numpy()
        camera_intrinsics = np.zeros_like(intrinsics)
        # camera_image = np.zeros((b*n*n_cam,self.condition_size[1],self.condition_size[0],3)).astype(np.uint8)
        camera_intrinsics[:,0] = intrinsics[:,0] * (self.condition_size[0] / self.rgb_size[0])
        camera_intrinsics[:,1] = intrinsics[:,1] * (self.condition_size[1] / self.rgb_size[1])
        camera_intrinsics[:,2] = intrinsics[:,2]
        category_dict = {1:'human',2:'vehicle'}
        imsize = (self.condition_size[0],self.condition_size[1])
        box_list = []
        camera_box_list = []
        for i in range(image.shape[0] * self.num_cameras):
            boxes = []
            idx = i // self.num_cameras
            for category in [1,2]:
                mask = (image[idx] == category)
                instance_label,_ = scipy.ndimage.label(mask.cpu().numpy())
                for j in range(1,_+1):
                    instance_mask = (instance_label == j)
                    if instance_mask.sum() == 0:
                        continue
                    y_indices,x_indices = np.where(instance_mask)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x_min,x_max,y_min,y_max = x_indices.min(),x_indices.max(),y_indices.min(),y_indices.max()
                        z_min = -0.15
                        z_max = 2 if category == 1 else 2.75
                        corners = []
                        for z in [z_min,z_max]:
                            for x,y in [(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)]:
                                corners.append([y,x,z])
                        corners = np.array(corners).astype(np.float64).transpose(1,0)
                        corners[0,:] = -(corners[0,:] - self.bev_size) * self.bev_resolution
                        corners[1,:] = -(corners[1,:] - self.bev_size // 2) * self.bev_resolution
                        yaw = Quaternion(global2ego_rotation[idx]).yaw_pitch_roll[0]
                        corners = np.dot(Quaternion(scalar=np.cos(yaw/2),vector=[0,0,np.sin(yaw/2)]).rotation_matrix,corners)
                        corners = np.dot(Quaternion(global2ego_rotation[idx]).rotation_matrix.T,corners)
                        corners = corners - translation[i]
                        corners = np.dot(rotation[i].T,corners)
                        # draw_box_in_camera_view(camera_image[i],corners,camera_intrinsics[i],imsize)
                        corners = self.get_box_in_image(corners,camera_intrinsics[i],imsize)
                        if corners is not None:
                            description = f"There is a annotation about {category_dict[category]},the center of callout box is ({np.mean(corners[0]):.2f},{np.mean(corners[1]):.2f})"
                            boxes.append(description)
            if len(boxes) > self.num_boxes:
                boxes = boxes[:self.num_boxes]
            elif len(boxes) == 0:
                boxes = ['None' for i in range(self.num_boxes)]
            else:
                boxes_none = ['None' for i in range(self.num_boxes - len(boxes))]
                boxes.extend(boxes_none)
            if n_cam == 1:
                box_list.append(boxes)
            else:
                camera_box_list.append(boxes)
                if i != 0 and i % n_cam == 0:
                    box_list.append(camera_box_list)
            # temp = Image.fromarray(camera_image[i])
            # temp.save(f"all_pics/condition/box_{i%n_cam}.png")
        if n_cam == 6 and camera_box_list != []:
            box_list.append(camera_box_list)
        return box_list

    def get_k_nearest_points(self,points,k=2):
        # points (n,2)
        if points.shape[0] < 2:
            return []
        kdtree = KDTree(points)
        neighbors = []
        distances,indices = kdtree.query(points,k=min(k+1,points.shape[0]),p=1)
        for i in range(points.shape[0]):
            neighbors.append(indices[i][1:k+1].tolist())
        return neighbors

    def get_hdmap(self,image,translation,rotation,intrinsics,global2ego_rotation):
        b,n,h,w = image.shape
        n_cam = self.num_cameras
        image = image.view(b*n,h,w)
        category_dict = {3:(0,0,255),4:(0,255,0),5:(255,0,0)}
        camera_image = np.zeros((b*n*n_cam,self.condition_size[1],self.condition_size[0],3)).astype(np.uint8)
        translation = translation.view(b*n*n_cam,4,1)[:,:3].cpu().numpy()
        rotation = rotation.view(b*n*n_cam,3,3).cpu().numpy()
        intrinsics = intrinsics.view(b*n*n_cam,3,3).cpu().numpy()
        global2ego_rotation = global2ego_rotation.view(b*n,4).cpu().numpy()
        camera_intrinsics = np.zeros_like(intrinsics)
        #TODO:cancel
        camera_intrinsics[:,0] = intrinsics[:,0] * (self.condition_size[0] / self.rgb_size[0])
        camera_intrinsics[:,1] = intrinsics[:,1] * (self.condition_size[1] / self.rgb_size[1])
        camera_intrinsics[:,2] = intrinsics[:,2]
        imsize = (self.condition_size[0],self.condition_size[1])
        for i in range(image.shape[0] * self.num_cameras):
            idx = i // self.num_cameras
            for category in [3,4,5]:
                mask = (image[idx] == category)
                positions = np.where(mask.cpu().numpy())
                positions = np.stack((positions[0],positions[1]),axis=-1)
                z = np.zeros((positions.shape[0],1))
                z[:,:] = -0.15
                positions = np.concatenate((positions,z),axis=-1)
                neighbors = self.get_k_nearest_points(positions)
                positions = positions.transpose(1,0)
                positions[0,:] = -(positions[0,:] - self.bev_size) * self.bev_resolution
                positions[1,:] = -(positions[1,:] - self.bev_size // 2) * self.bev_resolution
                yaw = Quaternion(global2ego_rotation[idx]).yaw_pitch_roll[0]
                positions = np.dot(Quaternion(scalar=np.cos(yaw/2),vector=[0,0,np.sin(yaw/2)]).rotation_matrix,positions)
                positions = np.dot(Quaternion(global2ego_rotation[idx]).rotation_matrix.T,positions)
                positions = positions - translation[i]
                positions = np.dot(rotation[i].T,positions)
                self.get_hdmap_in_image(camera_image[i],positions,camera_intrinsics[i],imsize,neighbors,category_dict[category])
        return camera_image

    def get_routemap(self,pred):
        b,n,h,w = pred.shape
        pred = pred.view(b*n,h,w)
        route_map = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2],3)).astype(np.uint8)
        label_color = {3:(255,0,0),4:(0,255,0),5:(0,0,255)}
        for i in range(pred.shape[0]):
            for label in [3,4,5]:
                mask = (pred[i] == label)
                positions = np.where(mask.cpu().numpy())
                positions = np.stack((positions[0],positions[1]),axis=-1)
                route_map[i,positions[:,0],positions[:,1]] = label_color[label]
        return route_map


    def forward(self,batch,output,):
        output = {k:v.cpu() for k,v in output.items() if isinstance(v,torch.Tensor)}
        batch = {k:v.cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        pred = torch.argmax(output['bev_segmentation_1'].detach(),dim=-3)
        translation = batch['ego2cam'][:,:,:,:,3:]
        rotation = batch['ego2cam'][:,:,:,:3,:3]
        intrinsics = batch['intrinsics']
        global2ego_rotation = rotation_6d_to_quaternion(output['orientation'])
        # global2ego_rotation = batch['global2ego']
        box_list = self.get_pred_box(pred,translation,rotation,intrinsics,global2ego_rotation)
        hdmap = self.get_hdmap(pred,translation,rotation,intrinsics,global2ego_rotation)
        route_map = self.get_routemap(pred)
        hdmap = torch.from_numpy(hdmap.transpose(0,3,1,2))
        route_map = torch.from_numpy(route_map.transpose(0,3,1,2))
        return pred,box_list,hdmap,route_map
        
        
