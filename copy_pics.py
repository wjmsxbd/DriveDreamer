import os
import glob
import json
import shutil
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AutoDM-training')
    parser.add_argument('--pic_path',
                        type=str,
                        help="pic save path")
    parser.add_argument('--resave_path',
                        type=str,
                        help="pic resave path")
    parser.add_argument('--json_path',
                        type=str,
                        help="pic resave path")
    cmd_args = parser.parse_args()
    pic_path = cmd_args.pic_path
    resave_path = cmd_args.resave_path
    json_path = cmd_args.json_path
    if not os.path.exists(resave_path):
        os.makedirs(resave_path)
    with open(json_path,'r') as file:
        map_idx2scene = json.load(file)
    all_pics = glob.glob(os.path.join(pic_path,'*.png'))
    for pic in all_pics:
        description = pic.split('/')[-1][:-4]
        first_frame_idx,frame_idx = description.split('_')
        first_frame_idx = str(int(first_frame_idx))
        frame_idx = int(frame_idx)
        scene_token = map_idx2scene[first_frame_idx]['token']
        scene_id = int(map_idx2scene[first_frame_idx]['scene_name'][-4:])
        dir_path = os.path.join(resave_path,scene_token)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        new_file_name = f'NUSCENES_{frame_idx:04d}.png'
        destination_file = os.path.join(dir_path,new_file_name)
        shutil.copy(pic,destination_file)
    


