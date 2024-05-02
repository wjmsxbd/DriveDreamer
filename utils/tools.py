import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
#from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
#from tools.analysis_tools.visualize.render.base_render import BaseRender
from nuscenes.map_expansion.map_api import NuScenesMap,NuScenesMapExplorer
from pyquaternion import Quaternion
import descartes
from shapely.geometry import Polygon,MultiPolygon,LineString,Point,box
import matplotlib.lines as mlines
import argparse
from nuscenes.nuscenes import NuScenes
import os,random
from torch.utils import data
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg

LOGGER_DEFAULT_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |'
                  ' <level>{level: <8}</level> |'
                  ' <cyan>{name}</cyan>:<cyan>{function}</cyan>:'
                  '<cyan>{line}</cyan> - <level>{message}</level>')

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def makepath(desired_path,isfile=True):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path):os.makedirs(desired_path)
    return desired_path

#TODO: box not in image also save
def get_box_in_image(box:Box,intrinsic:np.ndarray):
    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :].flatten()
    return corners_img

def get_3dbox(sample_data_token:str,nusc:NuScenes,imsize:tuple,out_path=None):
    data_path,boxes,camera_intrinsic = nusc.get_sample_data(sample_data_token,box_vis_level=True)
    #fig = plt.figure(figsize=(16,9),facecolor='black',dpi=100)
    # ax = fig.add_axes([0,0,1,1])
    # ax.set_xlim(0,imsize[0])
    # ax.set_ylim(0,imsize[1])
    box_category = []
    box_list = []
    for box in boxes:
        c = np.array([1.0,0.0,0.0])
        #box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
        box_category.append(box.name)
        box_list.append(get_box_in_image(box,camera_intrinsic))


    # ax.axis('off')
    # ax.set_aspect('equal')
    # ax.set_xlim(0, imsize[0])
    # ax.set_ylim(imsize[1], 0)
    # if out_path is not None:
    #     plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    # return fig,ax,box_category
    return box_list,box_category

def get_hdmap(sample_data_token:str,
              nusc:NuScenes,
              nusc_map:NuScenesMap,
              patch_radius:float = 10000,
              render_behind_cam:bool = True,
              render_outside_im:bool = True,
              min_polygon_area: float = 1000,
              outpath:str=None):
        layer_names = ['lane_divider','lane','ped_crossing']
        _,cs_record,pose_record,cam_intrinsic,imsize,yaw,translation = get_image_info(sample_data_token,nusc)
        im = Image.open(_)
        box_coords = (
            pose_record['translation'][0] - patch_radius,
            pose_record['translation'][1] - patch_radius,
            pose_record['translation'][0] + patch_radius,
            pose_record['translation'][1] + patch_radius,
        )
        near_plane = 1e-8
        records_in_patch = nusc_map.get_records_in_patch(box_coords,layer_names,mode='intersect')

        #Init axes.
        fig = plt.figure(figsize=(16,9),facecolor='black',dpi=100)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim(0,imsize[0])
        ax.set_ylim(0,imsize[1])
        
        layer_color = {'ped_crossing':'green','lane':'red'}
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_map.get(layer_name,token)
                if layer_name == 'lane_divider':
                    line_token = record['line_token']
                    line = nusc_map.extract_line(line_token)
                    points = np.array(line.xy)
                    points = np.vstack((points,np.zeros((1,points.shape[1]))))
                    points = points - np.array(pose_record['translation']).reshape(-1,1)
                    points = np.dot(Quaternion(pose_record['rotation']).rotation_matrix.T,points)
                    points = points - np.array(cs_record['translation']).reshape(-1,1)
                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T,points)
                    depths = points[2,:]
                    behind = depths < near_plane
                    if np.all(behind):
                        continue
                    if render_behind_cam:
                        points = NuScenesMapExplorer._clip_points_behind_camera(points,near_plane)
                    elif np.any(behind):
                        continue
                    if len(points) == 0 or points.shape[1]<2:
                        continue
                    points = view_points(points,cam_intrinsic,normalize=True)
                    # Skip polygons where all points are outside the image.
                    # Leave a margin of 1 pixel for aesthetic reasons.
                    inside = np.ones(points.shape[1],dtype=bool)
                    inside = np.logical_and(inside,points[0,:]>1)
                    inside = np.logical_and(inside,points[0,:]<imsize[0] - 1)
                    inside = np.logical_and(inside,points[1,:]>1)
                    inside = np.logical_and(inside,points[1,:]< imsize[1]-1)
                    if render_outside_im:
                        if np.all(np.logical_not(inside)):
                            continue
                    else:
                        if np.any(np.logical_not(inside)):
                            continue
                    line_proj = mlines.Line2D(points[0,:],points[1,:],color='blue')
                    ax.add_line(line_proj)
                else:
                    polygon_tokens = [record['polygon_token']]
                    for polygon_token in polygon_tokens:
                        polygon = nusc_map.extract_polygon(polygon_token)

                        # Convert polygon nodes to pointcloud with 0 height.
                        points = np.array(polygon.exterior.xy)
                        points = np.vstack((points,np.zeros((1,points.shape[1]))))

                        # Transform into the ego vehicle frame for the timestamp of the image.
                        points = points - np.array(pose_record['translation']).reshape(-1,1)
                        points = np.dot(Quaternion(pose_record['rotation']).rotation_matrix.T,points)

                        # Transform into the camera.
                        points = points - np.array(cs_record['translation']).reshape(-1,1)
                        points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T,points)

                         # Remove points that are partially behind the camera.
                        depths = points[2,:]
                        behind = depths < near_plane
                        if np.all(behind):
                            continue
                        if render_behind_cam:
                            points = NuScenesMapExplorer._clip_points_behind_camera(points,near_plane)
                        elif np.any(behind):
                            continue

                        if len(points) == 0 or points.shape[1] < 3:
                            continue

                        points = view_points(points,cam_intrinsic,normalize=True)
                        # Skip polygons where all points are outside the image.
                        # Leave a margin of 1 pixel for aesthetic reasons.
                        inside = np.ones(points.shape[1],dtype=bool)
                        inside = np.logical_and(inside,points[0,:]>1)
                        inside = np.logical_and(inside,points[0,:]<imsize[0] - 1)
                        inside = np.logical_and(inside,points[1,:]>1)
                        inside = np.logical_and(inside,points[1,:]< imsize[1]-1)
                        if render_outside_im:
                            if np.all(np.logical_not(inside)):
                                continue
                        else:
                            if np.any(np.logical_not(inside)):
                                continue

                        points = points[:2,:]
                        points = [(p0,p1) for (p0,p1) in zip(points[0],points[1])]
                        polygon_proj = Polygon(points)

                        if polygon_proj.area < min_polygon_area:
                            continue
                        label = layer_name
                        
                        plt.plot([point[0] for point in points],[point[1] for point in points],color=layer_color[layer_name])
                        #ax.add_patch(descartes.PolygonPatch(polygon_proj,fc=nusc_map.explorer.color_map[layer_name],ec='red',alpha=0,label=label))
        plt.axis('off')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        if outpath is not None:
            plt.savefig(outpath,bbox_inches='tight',pad_inches=0)
        return fig,ax,yaw,translation

def get_image_info(sample_data_token:str,
                   nusc:NuScenes):
    sd_record = nusc.get('sample_data',sample_data_token)
    cs_record = nusc.get('calibrated_sensor',sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor',cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose',sd_record['ego_pose_token'])
    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    translation = np.array(pose_record['translation'])
    data_path = nusc.get_sample_data_path(sample_data_token)
    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'],sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    return data_path,cs_record,pose_record,cam_intrinsic,imsize,yaw,translation

def convert_fig_to_numpy(fig,imsize):
    canvas = FigureCanvasAgg(fig)    
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, np.uint8).reshape(imsize[1],imsize[0],4)
    return img

def get_this_scene_info(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str):
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    cam_front_path = os.path.join(dataset_dir,cam_front_path)
    cam_front_img = mpimg.imread(cam_front_path)
    #mpimg.imsave(f'./temp/camera_front/{count:02d}.jpg',cam_front_img)
    imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
    box_list,box_category = get_3dbox(cam_front_token,nusc,imsize)#out_path=f'./temp/3dbox/{count:02d}.jpg'
    hdmap_fig,hdmap_ax,yaw,translation = get_hdmap(cam_front_token,nusc,nusc_map)#,outpath=f'./temp/hdmap/{count:02d}.jpg'
    now_hdmap = convert_fig_to_numpy(hdmap_fig,imsize)
    box_list = np.array(box_list)
    plt.close(hdmap_fig)
    return cam_front_img,box_list,now_hdmap,box_category,yaw,translation
    