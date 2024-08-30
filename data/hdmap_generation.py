import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import logging
from utils.tools import get_hdmap
import omegaconf
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

def process_hdmap(idx,nusc,nusc_maps,data_infos,sample_path,sweeps_path):
    print(f"now:pid:{os.getpid()},idx:{idx}")
    data = data_infos[idx]
    sample_token = data['token']
    sample_record = nusc.get('sample',sample_token)
    scene_token = sample_record['scene_token']
    scene = nusc.get('scene',scene_token)
    log_token = scene['log_token']
    log = nusc.get('log',log_token)
    nusc_map = nusc_maps[log['location']]
        
    cam_front_token = sample_record['data']['CAM_FRONT']
    cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    print(cam_front_path)
    outpath = cam_front_path.split('/')[-1][:-4]
    outpath = outpath + '.png'
    type = cam_front_path.split('/')[0]
    if type == 'samples':
        outpath = os.path.join(sample_path,outpath)
    else:
         outpath = os.path.join(sweeps_path,outpath)
    print(outpath)
    get_hdmap(cam_front_token,nusc,nusc_map,outpath=outpath)

def process_data(cfg):
    nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
    nusc_maps = {
        'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
        'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
        'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
        'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
    }
    split_name = cfg['split_name']
    data_info_path = os.path.join(cfg['dataroot'],f"nuScenes_advanced_infos_{split_name}.pkl")
    with open(data_info_path,'rb') as f:
        data_infos = pickle.load(f)
    data_infos = data_infos['infos']
    print(f"process len:{len(data_infos)}")
    sample_path = os.path.join(cfg['dataroot'],'samples_hdmap','CAM_FRONT')
    sweeps_path = os.path.join(cfg['dataroot'],'sweeps_hdmap','CAM_FRONT')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if not os.path.exists(sweeps_path):
        os.makedirs(sweeps_path)

    data_infos = list(sorted(data_infos,key=lambda e:e['timestamp']))
    # manager = multiprocessing.Manager()
    # progress = manager.Value('i',0)
    total_task = len(data_infos)
    f = partial(process_hdmap,nusc=nusc,nusc_maps=nusc_maps,data_infos=data_infos,sample_path=sample_path,sweeps_path=sweeps_path)
    # with ProcessPoolExecutor(max_workers=cfg['num_workers']) as executor:
    #     indices = range(len(data_infos))
    #     with tqdm(total=total_task) as pbar:
    #         for _ in executor.map(f,indices):
    #             pbar.update()
    # excutor = ProcessPoolExecutor(cfg['num_workers'])
    # for _,data in tqdm(enumerate(data_infos)):
    #     excutor.map(f,[_])
    indices = range(len(data_infos))
    with Pool(cfg['num_workers']) as p:
        # p.map(f,indices)
        p.map(f,indices)
    # for _,data in tqdm(enumerate(data_infos)):
    #     sample_token = data['token']
    #     sample_record = nusc.get('sample',sample_token)
    #     scene_token = sample_record['scene_token']
    #     scene = nusc.get('scene',scene_token)
    #     log_token = scene['log_token']
    #     log = nusc.get('log',log_token)
    #     nusc_map = nusc_maps[log['location']]
        
    #     cam_front_token = sample_record['data']['CAM_FRONT']
    #     cam_front_path = nusc.get('sample_data',cam_front_token)['filename']
    #     print(cam_front_path)
    #     outpath = cam_front_path.split('/')[-1][:-4]
    #     outpath = outpath + '.png'
    #     type = cam_front_path.split('/')[0]
    #     if type == 'samples':
    #         outpath = os.path.join(sample_path,outpath)
    #     else:
    #         outpath = os.path.join(sweeps_path,outpath)
    #     print(outpath)
    #     get_hdmap(cam_front_token,nusc,nusc_map,outpath=outpath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='hdmap generation')
    parser.add_argument('--config',
                        default='configs/hdmap_generation.yaml',
                        type=str,
                        help='config path')
    cmd_args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(cmd_args.config)

    process_data(cfg['train'])
    process_data(cfg['val'])