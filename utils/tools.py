import io

import torch
import cv2
import numpy as np
import os.path as osp
from PIL import Image,ImageDraw
import matplotlib.figure as mpfigure
import matplotlib.pyplot as plt
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import PointCloud, Box
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
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors,get_labels_in_coloring,create_lidarseg_legend,paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label,stuff_cat_ids
from typing import Tuple, List, Iterable
from ip_basic.ip_basic.depth_map_utils import fill_in_fast,fill_in_multiscale
# import open3d as o3d
from einops import rearrange
import matplotlib
import imageio
from io import BytesIO
from einops import repeat
# import objgraph
# from pympler import tracker,summary,muppy
import math
import cv2
try:
    import moxing as mox

    mox.file.shift('os', 'mox')
except:
    pass

eps = 1e-10
def sgn(x):
    if x < -eps:
        return -1
    if x > eps:
        return 1
    return 0

class Point:
    # TODO:modify dtype to float64
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y

    def __add__(self,other):
        return Point(self.x + other.x,self.y + other.y)

    def __sub__(self,other):
        return Point(self.x - other.x,self.y - other.y)

    def __mul__(self,other):
        if isinstance(other,Point):
            return self.x*other.x+self.y*other.y
        else:
            return Point(self.x * other,self.y * other)

    def __truediv__(self,other):
        return Point(self.x / other,self.y / other)

    def __eq__(self,other):
        return math.abs(self.x - other.x) <= eps and math.abs(self.y - other.y) <= eps
    
def cross(a:Point,b:Point):
    return a.x * b.y - a.y * b.x

def point_on_segment(p:Point,a:Point,b:Point):
    return sgn(cross(b - a,p - a)) == 0 and sgn((p-a) * (p-b)) <= 0

def has_intersection(a:Point,b:Point,p:Point,q:Point):
    d1,d2,d3,d4 = sgn(cross(b-a,p-a)),sgn(cross(b-a,q-a)),sgn(cross(q-p,a-p)),sgn(cross(q-p,b-p))
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 1 # have intersection and the intersection point is not at the endpoint
    if (d1 == 0 and point_on_segment(p,a,b)) or (d2==0 and point_on_segment(q,a,b)) or (d3==0 and point_on_segment(a,p,q)) or (d4==0 and point_on_segment(b,p,q)):
        return -1 # overlapping or intersection point is at the endpoint
    return 0

def line_intersection(a:Point,b:Point,p:Point,q:Point):
    # p!=q and a!=b
    U = cross(p-a,q-p)
    D = cross(b-a,q-p)
    if sgn(D) == 0:
        if sgn(U) == 0:
            return 2,None # overlapping
        else:
            return 0,None # parallel
    o = a + (b-a) * (U / D)
    return 1,o

def get_intersection_with_squre(a:Point,b:Point,x:Point,y:Point,z:Point,w:Point):
    # a->b x->y->z->w->x
    def is_better(A,B):
        if B is None:
            return True

    d1,d2,d3,d4 = has_intersection(a,b,x,y),has_intersection(a,b,y,z),has_intersection(a,b,z,w),has_intersection(a,b,w,x)
    candidate_points = []
    if d1 == 1:
        _,o = line_intersection(a,b,x,y)
        if _ == 1:
            candidate_points.append(o)
    elif d1 == -1:
        _,o = line_intersection(a,b,x,y)
        if _ == 2:
            candidate_points.append(o)

    if d2 == 1:
        _,o = line_intersection(a,b,y,z)
        if _ == 1:
            candidate_points.append(o)
    elif d2 == -1:
        _,o = line_intersection(a,b,y,z)
        if _ == 2:
            candidate_points.append(o)

    if d3 == 1:
        _,o = line_intersection(a,b,z,w)
        if _ == 1:
            candidate_points.append(o)
    elif d3 == -1:
        _,o = line_intersection(a,b,z,w)
        if _ == 2:
            candidate_points.append(o)

    if d4 == 1:
        _,o = line_intersection(a,b,w,x)
        if _ == 1:
            candidate_points.append(o)
    elif d4 == -1:
        _,o = line_intersection(a,b,w,x)
        if _ == 2:
            candidate_points.append(o)

    return candidate_points

class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)
        if file_name.startswith('obs'):
            with mox.file.File(file_name, 'rb') as f:
                file_data = f.read()
            scan = np.frombuffer(file_data, dtype=np.float32)
        else:
            scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)


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
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    return corners_img

def get_3dbox(sample_data_token:str,nusc:NuScenes,imsize:tuple,out_path=None):
    data_path,boxes,camera_intrinsic = nusc.get_sample_data(sample_data_token,box_vis_level=True)
    # fig = plt.figure(figsize=(16,9),facecolor='black',dpi=100)
    # ax = fig.add_axes([0,0,1,1])
    # ax.set_xlim(0,imsize[0])
    # ax.set_ylim(0,imsize[1])
    box_category = []
    box_list = []
    for box in boxes:
        c = np.array([1.0,0.0,0.0])
        # box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
        box_category.append(box.name)
        box_xy = get_box_in_image(box,camera_intrinsic)
        box_xy[0] = box_xy[0] / imsize[0]
        box_xy[1] = box_xy[1] / imsize[1]
        box_xy = box_xy.flatten()
        box_list.append(box_xy)

    # ax.axis('off')
    # ax.set_aspect('equal')
    # ax.set_xlim(0, imsize[0])
    # ax.set_ylim(imsize[1], 0)
    # if out_path is not None:
    #     plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    # return fig,ax,box_category
    return box_list,box_category

def get_hdmap_with_fig(fig,
                        ax,
                        sample_data_token:str,
                        nusc:NuScenes,
                        nusc_map:NuScenesMap,
                        patch_radius:float = 10000,
                        render_behind_cam:bool = True,
                        render_outside_im:bool = True,
                        min_polygon_area: float = 1000,
                        outpath:str=None):
    layer_names = ['lane_divider','lane','ped_crossing']
    _,cs_record,pose_record,cam_intrinsic,imsize,yaw,translation = get_image_info(sample_data_token,nusc)
    # im = Image.open(_)
    box_coords = (
        pose_record['translation'][0] - patch_radius,
        pose_record['translation'][1] - patch_radius,
        pose_record['translation'][0] + patch_radius,
        pose_record['translation'][1] + patch_radius,
    )
    near_plane = 1e-8
    records_in_patch = nusc_map.get_records_in_patch(box_coords,layer_names,mode='intersect')
    ax.set_xlim(0,imsize[0])
    ax.set_ylim(0,imsize[1])

    layer_color = {'ped_crossing':'green','lane':'red'}
    for layer_name in layer_names:
        for token in records_in_patch[layer_name]:
            record = nusc_map.get(layer_name,token)
            if layer_name == 'lane_divider':
                line_token = record['line_token']
                line = nusc_map.extract_line(line_token)
                points = np.array(line.xy).copy()
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
                    points = np.array(polygon.exterior.xy).copy()
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
    hdmap = convert_fig_to_numpy3(fig)
    fig.clf()
    ax.cla()
    return hdmap

def get_hdmap(sample_data_token:str,
              nusc:NuScenes,
              nusc_map:NuScenesMap,
              patch_radius:float = 80,
              render_behind_cam:bool = True,
              render_outside_im:bool = True,
              outpath:str=None):
        def check_out_windows(point):
            return point.x < 0 or point.x >= 1600 or point.y < 0 or point.y >= 1200

        layer_names = ['lane','ped_crossing']
        # layer_names = ['lane','ped_crossing']
        _,cs_record,pose_record,cam_intrinsic,imsize,yaw,translation = get_image_info(sample_data_token,nusc)
        # im = Image.open(_)
        # im.save('test_.png')
        box_coords = (
            pose_record['translation'][0] - patch_radius,
            pose_record['translation'][1] - patch_radius,
            pose_record['translation'][0] + patch_radius,
            pose_record['translation'][1] + patch_radius,
        )
        near_plane = 1e-8
        records_in_patch = nusc_map.get_records_in_patch(box_coords,layer_names,mode='intersect')

        #Init axes.
        hdmap = np.zeros((1200,1600,3)).astype(np.uint8)
        left_up_corner = Point(0,0)
        left_down_corner = Point(0,1200-1)
        right_up_corner = Point(1600-1,0)
        right_down_corner = Point(1600-1,1200-1)
        layer_color = {'ped_crossing':(0,255,0),'lane':(0,0,255),'lane_divider':(255,0,0)}
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_map.get(layer_name,token)
                if layer_name == 'lane_divider':
                    line_token = record['line_token']
                    line = nusc_map.extract_line(line_token)
                    points = np.array(line.xy).copy()
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
                    points = points[:2]
                    for i in range(len(points[0])-1):
                        point_A,point_B = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                        Bool_A,Bool_B = check_out_windows(point_A),check_out_windows(point_B)
                        if Bool_A and Bool_B:
                            PointA,PointB = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                            pointC = get_intersection_with_squre(PointA,PointB,left_up_corner,right_up_corner,right_down_corner,left_down_corner)
                            if len(pointC) == 2:
                                PointA,PointB = pointC[0],pointC[1]
                                cv2.line(hdmap,np.array([PointA.x,PointA.y],dtype=np.int16),np.array([PointB.x,PointB.y],dtype=np.int16),color=layer_color[layer_name],thickness=3)
                        elif Bool_A or Bool_B:
                            PointA,PointB = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                            pointC = get_intersection_with_squre(PointA,PointB,left_up_corner,right_up_corner,right_down_corner,left_down_corner)
                            if pointC == []:
                                continue
                            if Bool_A:
                                PointA = pointC[0]
                            else:
                                PointB = pointC[0]
                            cv2.line(hdmap,np.array([PointA.x,PointA.y],dtype=np.int16),np.array([PointB.x,PointB.y],dtype=np.int16),color=layer_color[layer_name],thickness=3)
                        else:
                            cv2.line(hdmap,points[:,i].astype(np.int16),points[:,i+1].astype(np.int16),color=layer_color[layer_name],thickness=3)
                else:
                    polygon_tokens = [record['polygon_token']]
                    for polygon_token in polygon_tokens:
                        polygon = nusc_map.extract_polygon(polygon_token)

                        # Convert polygon nodes to pointcloud with 0 height.
                        points = np.array(polygon.exterior.xy).copy()
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
                        # if polygon_proj.area < min_polygon_area or polygon_proj.area > 1000000:
                        #     continue
                        points = points[:2,:]
                        for i in range(len(points[0])-1):
                            point_A,point_B = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                            Bool_A,Bool_B = check_out_windows(point_A),check_out_windows(point_B)
                            if Bool_A and Bool_B:
                                PointA,PointB = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                                pointC = get_intersection_with_squre(PointA,PointB,left_up_corner,right_up_corner,right_down_corner,left_down_corner)
                                if len(pointC) == 2:
                                    PointA,PointB = pointC[0],pointC[1]
                                    cv2.line(hdmap,np.array([PointA.x,PointA.y],dtype=np.int16),np.array([PointB.x,PointB.y],dtype=np.int16),color=layer_color[layer_name],thickness=3)
                            elif Bool_A or Bool_B:
                                PointA,PointB = Point(points[0,i],points[1,i]),Point(points[0,i+1],points[1,i+1])
                                pointC = get_intersection_with_squre(PointA,PointB,left_up_corner,right_up_corner,right_down_corner,left_down_corner)
                                if pointC == []:
                                    continue
                                if Bool_A:
                                    PointA = pointC[0]
                                else:
                                    PointB = pointC[0]
                                cv2.line(hdmap,np.array([PointA.x,PointA.y],dtype=np.int16),np.array([PointB.x,PointB.y],dtype=np.int16),color=layer_color[layer_name],thickness=3)
                            else:
                                cv2.line(hdmap,points[:,i].astype(np.int16),points[:,i+1].astype(np.int16),color=layer_color[layer_name],thickness=3)
        if outpath is not None:
            temp = Image.fromarray(hdmap)
            temp.save(outpath)
        return hdmap

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
    # canvas.close_event()
    return img
def convert_fig_to_numpy2(fig):
    import PIL.Image as Image
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(),dtype=np.uint8)
    buf.shape = (w,h,4)
    buf = np.roll(buf,3,axis=2)
    image = Image.frombytes("RGBA",(w,h),buf.tostring())
    image = np.asarray(image)
    return image
def convert_fig_to_numpy3(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    buf.close()
    return img_np

def get_this_scene_info(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448)):
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    cam_front_path = os.path.join(dataset_dir,cam_front_path)
    if cam_front_path.startswith('obs'):
        with mox.file.File(cam_front_path, 'rb') as f:
            img_data = f.read()
        cam_front_img = Image.open(io.BytesIO(img_data))
        imsize = cam_front_img.size
        cam_front_img = np.array(cam_front_img.resize(img_size))
    else:
        cam_front_img = mpimg.imread(cam_front_path)
        #mpimg.imsave(f'./temp/camera_front/{count:02d}.jpg',cam_front_img)
        imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
        print(imsize)
        cam_front_img = Image.fromarray(cam_front_img)
        cam_front_img = np.array(cam_front_img.resize(img_size))

    box_list,box_category = get_3dbox(cam_front_token,nusc,imsize)#out_path=f'./temp/3dbox/{count:02d}.jpg'
    hdmap_fig,hdmap_ax,yaw,translation = get_hdmap(cam_front_token,nusc,nusc_map)#,outpath=f'./temp/hdmap/{count:02d}.jpg'
    now_hdmap = convert_fig_to_numpy(hdmap_fig,imsize)
    now_hdmap = Image.fromarray(now_hdmap)
    now_hdmap = np.array(now_hdmap.resize(img_size))

    box_list = np.array(box_list)
    plt.close(hdmap_fig)
    return cam_front_img,box_list,now_hdmap,box_category,yaw,translation

def get_this_scene_info_with_lidar_figax(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448),return_camera_info=False,fig=None,ax=None):
    matplotlib.use("Agg")
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_calibrated_sensor_token = nusc.get('sample_data',cam_front_token)['calibrated_sensor_token']
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data',pointsensor_token)
    Lidar_TOP_record = nusc.get('calibrated_sensor',pointsensor['calibrated_sensor_token'])
    cam_front_record = nusc.get('calibrated_sensor',cam_front_calibrated_sensor_token)
    Lidar_TOP_poserecord = nusc.get('ego_pose',pointsensor['ego_pose_token'])
    cam_poserecord = nusc.get('ego_pose',nusc.get('sample_data',cam_front_token)['ego_pose_token'])

    # cam_front_translation = torch.tensor(cs_record['translation'])
    # cam_front_rotation = torch.tensor(cs_record['rotation'])

    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    cam_front_path = os.path.join(dataset_dir,cam_front_path)

    # mpimg.imsave(f'./temp/depth_camera_front/{sample_token}.png',depth_cam_front_img)
    cam_front_img = imageio.v2.imread(cam_front_path)
    imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
    cam_front_img = Image.fromarray(cam_front_img)
    cam_front_img = np.array(cam_front_img.resize(img_size))
    # project_to_image(nusc,sample_token,out_path="004.png")
    # box_list,box_category = get_3dbox(cam_front_token,nusc,imsize)#out_path=f'./temp/3dbox/{count:02d}.jpg'
    # box_list = np.array(box_list)
    box_list = np.random.randn(70,16)
    box_category = ["None" for i in range(70)]
    
    # nusc.render_pointcloud_in_image(sample_token,out_path="002.png")

    # range_image = project_to_image(nusc,sample_token)
    # dense_range_image,_ = fill_in_multiscale(range_image,max_depth=100)
    # range_image = np.repeat(range_image[:,:,np.newaxis],3,axis=-1)
    range_image = np.random.randn(128,256,3)
    dense_range_image = np.random.randn(128,256)

    now_hdmap = get_hdmap_with_fig(fig,ax,cam_front_token,nusc,nusc_map)#,outpath=f'./temp/hdmap/{count:02d}.jpg'

    now_hdmap = Image.fromarray(now_hdmap)
    now_hdmap = np.array(now_hdmap.resize(img_size),dtype=np.uint8)
    if return_camera_info == False:
        return cam_front_img,box_list,now_hdmap,box_category,range_image,dense_range_image
    else:
        return cam_front_img,box_list,now_hdmap,box_category,range_image,dense_range_image,cam_front_record,cam_poserecord,Lidar_TOP_record,Lidar_TOP_poserecord

# @profile(precision=4,stream=open('log.txt',"w+",encoding="utf-8"))
def get_this_scene_info_with_lidar(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448),return_camera_info=False,collect_data=None):
    all_data = {}
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_calibrated_sensor_token = nusc.get('sample_data',cam_front_token)['calibrated_sensor_token']
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data',pointsensor_token)
    # ego_pose = nusc.get('ego_pose',pointsensor['ego_pose_token'])
    # print(ego_pose['translation'])
    Lidar_TOP_record = nusc.get('calibrated_sensor',pointsensor['calibrated_sensor_token'])
    cam_front_record = nusc.get('calibrated_sensor',cam_front_calibrated_sensor_token)
    Lidar_TOP_poserecord = nusc.get('ego_pose',pointsensor['ego_pose_token'])
    cam_poserecord = nusc.get('ego_pose',nusc.get('sample_data',cam_front_token)['ego_pose_token'])
    # get_bev_hdmap(cam_front_token,nusc,nusc_map)
    if 'bev_images' in collect_data:
        # outpath = 'test.png'
        bev_images = render_ego_centric_map(cam_front_token,nusc,nusc_map,img_size=img_size)
        all_data['bev_images'] = bev_images
    
    if 'Lidar_TOP_record' in collect_data:
        all_data['Lidar_TOP_record'] = Lidar_TOP_record
    if 'Lidar_TOP_poserecord' in collect_data:
        all_data['Lidar_TOP_poserecord'] = Lidar_TOP_poserecord
    all_data['cam_front_record'] = cam_front_record
    if 'cam_poserecord' in collect_data:
        all_data['cam_poserecord'] = cam_poserecord

    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    cam_front_path = os.path.join(dataset_dir,cam_front_path)
    if cam_front_path.startswith('obs'):
        with mox.file.File(cam_front_path, 'rb') as f:
            img_data = f.read()
        cam_front_img = Image.open(io.BytesIO(img_data))
        imsize = cam_front_img.size
        cam_front_img = np.array(cam_front_img.resize(img_size))
        if 'reference_image' in collect_data:
            all_data['reference_image'] = cam_front_img
    else:
        cam_front_img = mpimg.imread(cam_front_path)
        imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
        cam_front_img = Image.fromarray(cam_front_img)
        cam_front_img = np.array(cam_front_img.resize(img_size))
        if 'reference_image' in collect_data:
            all_data['reference_image'] = cam_front_img

    if 'HDmap' in collect_data:
        hdmap = get_hdmap(cam_front_token,nusc,nusc_map)
        hdmap = Image.fromarray(hdmap)
        hdmap = np.array(hdmap.resize(img_size),dtype=np.uint8)
        all_data['HDmap'] = hdmap

    if '3Dbox' in collect_data:
        box_list,box_category = get_3dbox(cam_front_token,nusc,imsize)#out_path=f'./temp/3dbox/{count:02d}.jpg'
        box_list = np.array(box_list)
        all_data['3Dbox'] = box_list
        all_data['category'] = box_category
    
    if 'range_image' in collect_data:
        range_image = project_to_image(nusc,sample_token,img_size=(img_size[1],img_size[0]))
        range_image = np.repeat(range_image[:,:,np.newaxis],3,axis=-1)
        all_data['range_image'] = range_image
    if 'dense_range_image' in collect_data:
        tmp_img = range_image[:,:,0]
        dense_range_image,_ = fill_in_multiscale(tmp_img,max_depth=100)
        dense_range_image = np.repeat(dense_range_image[...,np.newaxis],3,axis=-1)
        all_data['dense_range_image'] = dense_range_image
    return all_data
def project_to_image(nusc: NuScenes,sample_token:str,pointsensor_channel: str='LIDAR_TOP',camera_channel:str='CAM_FRONT',out_path:str=None,
                     img_size=(128,256)):
    sample_record = nusc.get('sample',sample_token)
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]
    cam = nusc.get('sample_data',camera_token)
    pointsensor = nusc.get('sample_data',pointsensor_token)
    pcl_path = osp.join(nusc.dataroot,pointsensor['filename'])
    scan = np.fromfile(pcl_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :3].T
    r = np.sqrt(points[0]**2 + points[1]**2 + points[2]**2)
    phi = np.arctan2(points[0],points[1])
    theta = np.arcsin(points[2] / r)
    mask = np.logical_and(phi > -np.pi/6,phi < np.pi/6)
    points = points[:,mask]
    phi = phi[mask]
    r = r[mask]
    theta = theta[mask]

    h = img_size[0]
    w = img_size[1]
    theta_down = -np.pi / 6
    theta_up = np.pi / 12
    theta = np.clip(theta,theta_down,theta_up)
    r = r.astype(np.uint8)
    row_indices = (theta_up - theta) / (theta_up - theta_down) * h - 1
    col_indices = w * (1 + phi / (np.pi / 6)) / 2 - 1
    row_indices = row_indices.astype(np.int16)
    col_indices = col_indices.astype(np.int16)
    range_image = np.zeros((h,w)).astype(np.uint8)
    for i in range(len(row_indices)):

        range_image[row_indices[i],col_indices[i]] = max(range_image[row_indices[i],col_indices[i]],r[i].copy())
    # range_image = np.array(range_image * 255.).astype(np.uint8)
    scan,points = None,None
    range_image = range_image.copy()
    if out_path is not None:
        image = Image.fromarray(range_image)
        image.save(out_path)
    return range_image

def get_gt_lidar_point(nusc:NuScenes,sample_token:str,return_intensity=False):
    sample_record = nusc.get('sample',sample_token)
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data',pointsensor_token)
    pcl_path = osp.join(nusc.dataroot,pointsensor['filename'])
    scan = np.fromfile(pcl_path, dtype=np.float32)
    if return_intensity:
        points = scan.reshape((-1,5))[:,:4].T
    else:
        points = scan.reshape((-1, 5))[:, :3].T
    phi = np.arctan2(points[0],points[1])
    mask = np.logical_and(phi > -np.pi/6,phi < np.pi/6)
    points = points[:,mask]
    return points.T

def project_to_bev(nusc:NuScenes,sample_token:str,outpath:str,res=0.5):
    sample_record = nusc.get('sample',sample_token)
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    camera_token = sample_record['data']['CAM_FRONT']
    pointsensor = nusc.get('sample_data',pointsensor_token)
    pcl_path = osp.join(nusc.dataroot,pointsensor['filename'])
    scan = np.fromfile(pcl_path, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :3].T

    r = np.sqrt(points[0]**2 + points[1]**2 + points[2]**2)
    phi = np.arctan2(points[0],points[1])
    mask = np.logical_and(phi > -np.pi/6,phi < np.pi/6)
    points = points[:,mask]


    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]

    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    x_img -= np.min(x_img)
    y_img -= np.min(y_img)

    top = np.zeros([np.max(y_img)+1,np.max(x_img)+1,3],dtype=np.float32)
    top[y_img,x_img,0] = z_points
    top = (top / 3 * 255).astype(np.uint8)
    if outpath is not None:
        img = Image.fromarray(top)
        img.save(outpath)
    return top

def get_bev_hdmap(sample_data_token:str,
                  nusc:NuScenes,
                  nusc_map:NuScenesMap,
                  patch_radius: float=25.6,
                  resolution: int=10,
                  img_size: List=[256,128]
                  ):
    def crop_image(image:np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: int,) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px // 2)
        y_max = int(y_px + axes_limit_px // 2)
        crop_image = image[y_min:y_max,x_min:x_max].copy()
        return crop_image
    layer_names = ['lane_divider','lane','ped_crossing']
    sd_record = nusc.get('sample_data',sample_data_token)
    pose_record = nusc.get('ego_pose',sd_record['ego_pose_token'])
    patch_radius_scale = patch_radius * 5
    box_coords = (
        pose_record['translation'][0] - patch_radius_scale,
        pose_record['translation'][1] - patch_radius_scale / 2,
        pose_record['translation'][0] + patch_radius_scale,
        pose_record['translation'][1] + patch_radius_scale / 2,
    )

    records_in_path = nusc_map.get_records_in_patch(box_coords,layer_names,mode='intersect')
    layer_color = {'ped_crossing':(0,255,0),'lane':(0,0,255),'lane_divider':(255,0,0)}
    bev_hdmap = np.zeros((int(patch_radius_scale / 2 * resolution+1),int(patch_radius_scale * resolution+1),3)).astype(np.uint8)
    for layer_name in layer_names:
        for token in records_in_path[layer_name]:
            record = nusc_map.get(layer_name,token)
            if layer_name == 'lane_divider':
                line_token = record['line_token']
                line = nusc_map.extract_line(line_token)
                points = np.array(line.xy).copy()
                # points = ((points - np.array(pose_record['translation'][:2])[:,np.newaxis])) * resolution + np.array([bev_hdmap.shape[0] // 2,bev_hdmap.shape[1] //2])[:,np.newaxis]
                points_x = (points[0] - np.array(pose_record['translation'][0] )) * resolution + bev_hdmap.shape[1] // 2
                points_y = (np.array(pose_record['translation'][1] - points[1])) * resolution + bev_hdmap.shape[0] // 2
                points = np.concatenate([points_x[np.newaxis,:],points_y[np.newaxis,:]],axis=0)
                points = points.astype(np.int16)
                for i in range(len(points[0])-1):
                    cv2.line(bev_hdmap,points[:,i],points[:,i+1],color=layer_color[layer_name],thickness=3)
            else:
                polygon_tokens = [record['polygon_token']]
                for polygon_token in polygon_tokens:
                    polygon = nusc_map.extract_polygon(polygon_token)
                    points = np.array(polygon.exterior.xy).copy()
                    # points = ((points - np.array(pose_record['translation'][:2])[:,np.newaxis])) * resolution + np.array([bev_hdmap.shape[0] // 2,bev_hdmap.shape[1] //2])[:,np.newaxis]
                    points_x = (points[0] - np.array(pose_record['translation'][0])) * resolution + bev_hdmap.shape[1] // 2
                    points_y = (np.array(pose_record['translation'][1] - points[1])) * resolution + bev_hdmap.shape[0] // 2
                    points = np.concatenate([points_x[np.newaxis,:],points_y[np.newaxis,:]],axis=0)
                    points = points.astype(np.int32)
                    points = points.T.reshape(-1,1,2)
                    cv2.polylines(bev_hdmap,[points],isClosed=True,color=layer_color[layer_name],thickness=3)
        
    ypr_rad = Quaternion(pose_record['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    bev_hdmap = np.array(Image.fromarray(bev_hdmap).rotate(yaw_deg))     
    bev_hdmap = crop_image(bev_hdmap,bev_hdmap.shape[1]//2,bev_hdmap.shape[0]//2,patch_radius*resolution)
    bev_hdmap = Image.fromarray(bev_hdmap)
    bev_hdmap = bev_hdmap.resize(img_size)
    bev_hdmap = np.array(bev_hdmap)
    return bev_hdmap



def project_to_camera(range_image,point_sensor_record,cam_sensor_record,ego_pose_record,img_size=(128,256),min_dist=1.0,out_path=None):
    indices = np.nonzero(range_image)
    points = np.zeros(len(indices),3)
    theta_up = np.pi / 12
    theta_down = -np.pi / 6
    theta_res = (theta_up - theta_down) / img_size[0]
    phi_res = (np.pi / 3) / img_size[1]
    for idx in range(len(indices)):
        vi,ui = indices[idx]
        phi = ((1+ui)/img_size[1] - 1) * torch.pi / 6
        theta = theta_up - (theta_up - theta_down) * ((vi+1) - 1./2) / img_size[0]
        r = range_image[vi,ui]
        point_x = r * torch.cos(theta) * torch.sin(phi)
        point_y = r * torch.cos(theta) * torch.cos(phi)
        point_z = r * torch.sin(theta)
        points[idx] = np.array([point_x,point_y,point_z])
    points = np.ascontiguousarray(points.transpose(1,0))
    pc = LidarPointCloud(points=points)
    pc.rotate(Quaternion(point_sensor_record['rotation']).rotation_matrix)
    pc.translate(np.array(point_sensor_record['translation']))
    
    
    pc.rotate(Quaternion(ego_pose_record['rotation']).rotation_matrix)
    pc.translate(np.array(ego_pose_record['translation']))
    
    pc.translate(-np.array(ego_pose_record['translation']))
    pc.rotate(Quaternion(ego_pose_record['rotation']).rotation_matrix.T)
    
    pc.translate(-np.array(cam_sensor_record['translation']))
    pc.rotate(Quaternion(cam_sensor_record['rotation']).rotation_matrix.T)
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cam_sensor_record['camera_intrinsic']), normalize=True)
    mask = np.ones(points[2,:].shape[0], dtype=bool)
    mask = np.logical_and(mask, points[2,:] > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < 1600 - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < 900 - 1)
    points = points[:, mask]
    cam_range_image = np.zeros((900,1600,3))
    
    for point in points:
        cam_range_image[point[1],point[0]] = point[2]
    cam_range_image = Image.fromarray(cam_range_image)
    if out_path is not None:
        cam_range_image.save(out_path)
    cam_range_image = np.array(cam_range_image.resize(img_size+(3,)))
    return cam_range_image


def render_pointcloud_in_image(nusc: NuScenes,sample_token:str,dot_size:int = 1,pointsensor_channel: str='LIDAR_TOP',
                               camera_channel:str='CAM_FRONT',out_path:str=None,
                               render_intensity:bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: bool = False,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = False,
                               lidarseg_preds_bin_path: str = None,
                               show_panoptic: bool = False) -> None:
    """
        Scatter-plots a pointcloud on top of image.
        :param sample_token: Sample token.
        :param dot_size: Scatter plot dot size.
        :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param ax: Axes onto which to render.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param verbose: Whether to display the image in a window.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
    if show_lidarseg:
        show_panoptic = False
    sample_record = nusc.get('sample',sample_token)
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    points,coloring,im = map_pointcloud_to_image(nusc,pointsensor_token,camera_token,
                                                 render_intensity=render_intensity,
                                                 show_lidarseg=show_lidarseg,
                                                 filter_lidarseg_labels=filter_lidarseg_labels,
                                                 lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                 show_panoptic=show_panoptic)
    # Init axes.
    fig = plt.figure(figsize=(16,9),facecolor='black',dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,im.size[0])
    ax.set_ylim(0,im.size[1])
    fig.set_facecolor('black')
    if lidarseg_preds_bin_path:
        fig.canvas.set_window_title(sample_token + "(prediction)")
    else:
        fig.canvas.set_window_title(sample_token)
    
    # ax.imshow(im)
    coloring = coloring / 255.
    
    coloring = np.repeat(coloring[:,np.newaxis],3,axis=1)
    # points[0,:] = (points[0,:] - points[0,:].min()) / (points[0,:].max() - points[0,:].min() + 1) * im.size[0]
    # points[1,:] = (points[1,:] - points[1,:].min()) / (points[1,:].max() - points[1,:].min() + 1) * im.size[1]
    ax.scatter(points[0,:],points[1,:],c=coloring,s=dot_size,marker='s')
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if pointsensor_channel == 'LIDAR_TOP' and (show_lidarseg or show_panoptic) and show_lidarseg_legend:
        # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
        # after it has been projected onto the image; this will allow displaying the legend only for classes which
        # are present in the image (instead of all the classes).
        if filter_lidarseg_labels is None:
            if show_lidarseg:
                # Since the labels are stored as class indices, we get the RGB colors from the
                # colormap in an array where the position of the RGB color corresponds to the index
                # of the class it represents.
                color_legend = colormap_to_colors(nusc.colormap,nusc.lidarseg_name2idx_mapping)
                filter_lidarseg_labels = get_labels_in_coloring(color_legend,coloring)
            else:
                filter_lidarseg_labels = stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping))

        if filter_lidarseg_labels and show_panoptic:
            stuff_labels = set(stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping)))
            filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

        create_lidarseg_legend(filter_lidarseg_labels,nusc.lidarseg_name2idx_mapping,nusc.colormap,
                               loc='upper left',ncol=1,bbox_to_anchor=(1.05,1.0))
    if out_path is not None:
        plt.savefig(out_path,bbox_inches='tight',pad_inches=0)
    if verbose:
        plt.show()
    return fig,ax


def map_pointcloud_to_image(nusc:NuScenes,
                            pointsensor_token:str,
                            camera_token:str,
                            min_dist:float=1.0,
                            render_intensity:bool=False,
                            show_lidarseg:bool=False,
                            filter_lidarseg_labels:List=None,
                            lidarseg_preds_bin_path:str=None,
                            show_panoptic:bool=False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    cam = nusc.get('sample_data',camera_token)
    pointsensor = nusc.get('sample_data',pointsensor_token)
    pcl_path = osp.join(nusc.dataroot,pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        if show_lidarseg or show_panoptic:
            gt_from = 'lidarseg' if show_lidarseg else 'panopic'
            assert hasattr(nusc,gt_from),f'Error: nuScenes-{gt_from} not installed!'

            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
            assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                             'render_intensity or show_lidarseg, not both.'
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = LidarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot,cam['filename']))
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose',pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose',cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    
    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor',cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2,:] 
    
    # pcd = o3d.geometry.PointCloud()
    # pcd_points = rearrange(pc.points[:3,],"c h -> h c")
    # pcd.points = o3d.utility.Vector3dVector(pcd_points)
    # crop h
    # points = pc.points[:3]
    # angles = np.arctan2(points[0],points[2])
    # min_angles = -np.pi / 6
    # max_angles = np.pi / 6
    # indices = np.where((angles>=min_angles) & (angles<=max_angles))[0]
    # points = points[:,indices]
    # #crop w
    # angles = np.arctan2(points[1],(points[0]**2+points[2]**2)**0.5)
    # min_angles = -np.pi / 18
    # max_angles = np.pi / 6
    # indices = np.where((angles>=min_angles) & (angles<=max_angles))[0]
    # points = points[:,indices]
    # coloring = points[2,:]
    # depths = pc.points[2,:] 
    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                              'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3,:]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0,intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                              'not %s!' % pointsensor['sensor_modality']
        gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
        semantic_table = getattr(nusc, gt_from)
        if lidarseg_preds_bin_path:
            sample_token = nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(semantic_table) > 0: # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(nusc.dataroot,
                                                    nusc.get(gt_from, pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None
        if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(lidarseg_labels_filename,
                                                filter_lidarseg_labels,
                                                nusc.lidarseg_name2idx_mapping,
                                                nusc.colormap)
            else:
                coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                    filter_lidarseg_labels,
                                                    nusc.lidarseg_name2idx_mapping,
                                                    nusc.colormap)
        else:
            coloring = depths
            print(f'Warning: There are no lidarseg labels in {nusc.version}. Points will be colored according '
                    f'to distance from the ego vehicle instead.')
    else:
        # Retrieve the color from the depth.
        coloring = depths
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    return points, coloring, im


def render_ego_centric_map(sample_data_token:str,
                           nusc:NuScenes,
                           nusc_map:NuScenesMap,
                           axes_limit: float = 25.6,
                           img_size: tuple = (256,128),
                           outpath = None):
    def crop_image(image:np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: int,) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px // 2)
        y_max = int(y_px + axes_limit_px // 2)
        crop_image = image[y_min:y_max,x_min:x_max].copy()
        return crop_image
    def check_coords(x,y,x_min,x_max,y_min,y_max):
        return x>=x_min and x<=x_max and y>=y_min and y<=y_max
    
    num_colors = 40
    colors = plt.cm.get_cmap('tab20', num_colors)
    colormap = {}
    idx = 0
    for category in nusc.category:
        colormap[category['name']] = colors.colors[idx][:3]
        idx += 1
    sd_record = nusc.get('sample_data',sample_data_token)
    sample = nusc.get('sample',sd_record['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    log = nusc.get('log',scene['log_token'])

    # map_path = os.path.join(nusc_map.dataroot,'maps/basemap',str(nusc_map.map_name)+'.png')
    # detail_map = mpimg.imread(map_path)

    map_ = nusc.get('map',log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose',sd_record['ego_pose_token'])
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0],pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    # mask_raster = Image.fromarray(mask_raster)
    # mask_raster.save('test.png')
    cropped = crop_image(mask_raster,pixel_coords[0],pixel_coords[1],int(scaled_limit_px * math.sqrt(5)))
    boxes = nusc.get_boxes(sample_data_token)
    # _,boxes,_ = nusc.get_sample_data(sample_data_token,use_flat_vehicle_coordinates=True)
    axes_limit_px = int(scaled_limit_px * math.sqrt(5))
    x_min = pixel_coords[0] - axes_limit_px
    x_max = pixel_coords[0] + axes_limit_px
    y_min = pixel_coords[1] - axes_limit_px // 2
    y_max = pixel_coords[1] + axes_limit_px // 2
    cropped[cropped == map_mask.foreground] = 128
    cropped[cropped == map_mask.background] = 255
    cropped[cropped.shape[0]//2-5:cropped.shape[0]//2+5,cropped.shape[1]//2-5:cropped.shape[1]//2+5] = 0
    cropped = np.repeat(cropped[...,np.newaxis],repeats=3,axis=-1)
    cropped = (cropped / 255.).astype(np.float32)
    for box in boxes:
        corners = box.corners()
        corners = corners.mean(axis=1)
        box_coords = map_mask.to_pixel_coords(corners[0],corners[1])
        if check_coords(box_coords[0],box_coords[1],x_min,x_max,y_min,y_max):
            box_coords = (box_coords[1][0] - y_min[0],box_coords[0][0] - x_min[0])
            box_colors = colormap[box.name]
            cropped[box_coords[0]-5:box_coords[0]+5,box_coords[1]-5:box_coords[1]+5] = box_colors
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    cropped = (cropped * 255.).astype(np.uint8)
    # rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
    rotated_cropped = cropped
    ego_centric_map = crop_image(rotated_cropped,
                                 int(rotated_cropped.shape[1] / 2),
                                 int(rotated_cropped.shape[0] / 2),
                                 scaled_limit_px)
    ego_centric_map = Image.fromarray(ego_centric_map).resize(img_size)
    ego_centric_map = np.array(ego_centric_map)
    outpath="test1.png"
    if outpath:
        img = Image.fromarray(ego_centric_map)
        img.save(outpath)
    return ego_centric_map
    
def render_ego_centric_map_detail(sample_data_token:str,
                           nusc:NuScenes,
                           nusc_map:NuScenesMap,
                           axes_limit: float = 25.6,
                           img_size: tuple = (256,128),
                           outpath = None):
    def crop_image(image:np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: int,) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px // 2)
        y_max = int(y_px + axes_limit_px // 2)
        crop_image = image[y_min:y_max,x_min:x_max].copy()
        return crop_image
    def check_coords(x,y,x_min,x_max,y_min,y_max):
        return x>=x_min and x<=x_max and y>=y_min and y<=y_max
    
    num_colors = 40
    colors = plt.cm.get_cmap('tab20', num_colors)
    colormap = {}
    idx = 0
    for category in nusc.category:
        colormap[category['name']] = colors.colors[idx][:3]
        idx += 1
    sd_record = nusc.get('sample_data',sample_data_token)
    sample = nusc.get('sample',sd_record['sample_token'])
    scene = nusc.get('scene',sample['scene_token'])
    log = nusc.get('log',scene['log_token'])

    map_path = os.path.join(nusc_map.dataroot,'maps/basemap',str(nusc_map.map_name)+'.png')
    detail_map = mpimg.imread(map_path)

    map_ = nusc.get('map',log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose',sd_record['ego_pose_token'])
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0],pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))

    cropped = crop_image(detail_map,pixel_coords[0],pixel_coords[1],int(scaled_limit_px * math.sqrt(5)))
    del detail_map
    boxes = nusc.get_boxes(sample_data_token)
    # _,boxes,_ = nusc.get_sample_data(sample_data_token,use_flat_vehicle_coordinates=True)
    axes_limit_px = int(scaled_limit_px * math.sqrt(5))
    x_min = pixel_coords[0] - axes_limit_px
    x_max = pixel_coords[0] + axes_limit_px
    y_min = pixel_coords[1] - axes_limit_px // 2
    y_max = pixel_coords[1] + axes_limit_px // 2
    
    cropped[cropped.shape[0]//2-5:cropped.shape[0]//2+5,cropped.shape[1]//2-5:cropped.shape[1]//2+5] = 0
    cropped = np.repeat(cropped[...,np.newaxis],repeats=3,axis=-1)
    for box in boxes:
        corners = box.corners()
        corners = corners.mean(axis=1)
        box_coords = map_mask.to_pixel_coords(corners[0],corners[1])
        if check_coords(box_coords[0],box_coords[1],x_min,x_max,y_min,y_max):
            box_coords = (box_coords[1][0] - y_min[0],box_coords[0][0] - x_min[0])
            box_colors = colormap[box.name]
            cropped[box_coords[0]-5:box_coords[0]+5,box_coords[1]-5:box_coords[1]+5] = box_colors
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    cropped = (cropped * 255.).astype(np.uint8)
    print(cropped.shape)
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
    ego_centric_map = crop_image(rotated_cropped,
                                 int(rotated_cropped.shape[1] / 2),
                                 int(rotated_cropped.shape[0] / 2),
                                 scaled_limit_px)
    ego_centric_map = Image.fromarray(ego_centric_map).resize(img_size)
    ego_centric_map = np.array(ego_centric_map)
    if outpath:
        img = Image.fromarray(ego_centric_map)
        img.save(outpath)
    return ego_centric_map

def get_depth_min_max(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448)):
    sample_record = nusc.get('sample',sample_token)
    range_image_fig,range_image_ax = render_pointcloud_in_image(nusc,sample_token)#out_path='000.jpg'
    range_image = convert_fig_to_numpy2(range_image_fig)
    dense_range_image = range_image[:,:,0]
    plt.close(range_image_fig)
    # return cam_front_img,box_list,now_hdmap,box_category,depth_cam_front_img,range_image,dense_range_image
    return dense_range_image.max(),dense_range_image.min()


if __name__ == "__main__":
    # o3d.io.read_point_cloud("/storage/group/4dvlab/datasets/nuScenes/mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin  n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448744447639.pcd.bin")
    pass