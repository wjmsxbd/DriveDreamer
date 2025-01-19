import numpy as np
import torch

def bev_params_to_intrinsics(size,scale,offsetx):
    """
        size: number of pixels (width, height)
        scale: pixel size (in meters)
        offsetx: offset in x direction (direction of car travel)
    """
    intrinsics_bev = np.array([
        [1/scale,0,size[0]/2+offsetx],
        [0,-1/scale,size[1]/2],
        [0,0,1],
    ],dtype=np.float32)
    return intrinsics_bev

def bev_6views_params_to_intrinsics(size,scale,offsetx,offsety):
    intrinsics_bev = np.array([
        [1/scale,0,size[0]/2+offsetx],
        [0,-1/scale,size[1]/2+offsety],
        [0,0,1]
    ],dtype=np.float32)
    return intrinsics_bev

def intrinsics_inverse(intrinsics):
    fx = intrinsics[...,0,0]
    fy = intrinsics[...,1,1]
    cx = intrinsics[...,0,2]
    cy = intrinsics[...,1,2]
    one = torch.ones_like(fx)
    zero = torch.zeros_like(fx)
    intrinsics_inv = torch.stack((
        torch.stack((1/fx, zero, -cx/fx), -1),
        torch.stack((zero, 1/fy, -cy/fy), -1),
        torch.stack((zero, zero, one), -1),
    ), -2)
    return intrinsics_inv

