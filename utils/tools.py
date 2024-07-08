import torch
import cv2
import numpy as np
import os.path as osp
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
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors,get_labels_in_coloring,create_lidarseg_legend,paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label,stuff_cat_ids
from typing import Tuple, List, Iterable

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

def get_this_scene_info(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448)):
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    cam_front_path = os.path.join(dataset_dir,cam_front_path)
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
    

def get_this_scene_info_with_lidar(dataset_dir,nusc:NuScenes,nusc_map:NuScenesMap,sample_token:str,img_size:tuple=(768,448)):
    sample_record = nusc.get('sample',sample_token)
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    depth_cam_front_path = cam_front_path.split('/')
    depth_cam_front_path[0] = depth_cam_front_path[0] + '_depth'
    depth_cam_front_path[-1] = depth_cam_front_path[-1].split('.')[0] + '_depth.png'
    cam_front_path = os.path.join(dataset_dir,cam_front_path)
    depth_cam_front_path = os.path.join(dataset_dir,*depth_cam_front_path)

    depth_cam_front_img = mpimg.imread(depth_cam_front_path).astype(np.float32)
    # mpimg.imsave(f'./temp/depth_camera_front/{sample_token}.png',depth_cam_front_img)
    cam_front_img = mpimg.imread(cam_front_path)
    #mpimg.imsave(f'./temp/camera_front/{count:02d}.jpg',cam_front_img)
    imsize = (cam_front_img.shape[1],cam_front_img.shape[0])
    cam_front_img = Image.fromarray(cam_front_img)
    cam_front_img = np.array(cam_front_img.resize(img_size))
    depth_cam_front_img = (depth_cam_front_img * 255.).astype(np.uint8)
    #print(depth_cam_front_img.min(),depth_cam_front_img.max())
    depth_cam_front_img = Image.fromarray(depth_cam_front_img)
    depth_cam_front_img = np.array(depth_cam_front_img.resize(img_size))

    

    box_list,box_category = get_3dbox(cam_front_token,nusc,imsize)#out_path=f'./temp/3dbox/{count:02d}.jpg'
    hdmap_fig,hdmap_ax,yaw,translation = get_hdmap(cam_front_token,nusc,nusc_map)#,outpath=f'./temp/hdmap/{count:02d}.jpg'
    range_image_fig,range_image_ax = render_pointcloud_in_image(nusc,sample_token)#out_path='000.jpg'
    range_image = convert_fig_to_numpy(range_image_fig,imsize)
    range_image = Image.fromarray(range_image)
    range_image = np.array(range_image.resize(img_size))

    now_hdmap = convert_fig_to_numpy(hdmap_fig,imsize)
    now_hdmap = Image.fromarray(now_hdmap)
    now_hdmap = np.array(now_hdmap.resize(img_size))

    box_list = np.array(box_list)
    plt.close(hdmap_fig)
    plt.close(range_image_fig)
    return cam_front_img,box_list,now_hdmap,box_category,depth_cam_front_img,range_image

def render_pointcloud_in_image(nusc: NuScenes,sample_token:str,dot_size:int = 5,pointsensor_channel: str='LIDAR_TOP',
                               camera_channel:str='CAM_FRONT',out_path:str=None,
                               render_intensity:bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: bool = False,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
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
    fig,ax = plt.subplots(1,1,figsize=(16,9))
    if lidarseg_preds_bin_path:
        fig.canvas.set_window_title(sample_token + "(prediction)")
    else:
        fig.canvas.set_window_title(sample_token)
    
    # ax.imshow(im)
    ax.scatter(points[0,:],points[1,:],c=coloring,s=dot_size)
    ax.axis('off')

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
        plt.savefig(out_path,bbox_inches='tight',pad_inches=0,dpi=200)
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
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose',pointsensor['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor',cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2,:]

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
