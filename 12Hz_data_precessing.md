# Prepare nuScenes-12Hz_data
Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). 
# Clone the git repository
```
git clone https://github.com/open-mmlab/mmdetection3d.git-b dev-1.0
git clone https://github.com/JeffWang987/ASAP.git
```
Ensure the two repository is under the same folder
# Create the environment for mmdetection3d
```
cd mmdetection3d
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.6.0
mim install mmdet==2.25.3
pip install -v -e .
```
Get the pretrain model from here https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth
# Create a conda virtual environment for ASAP


**1. Create a conda virtual environment for ASAP**
```shell
conda create -n ASAP python=3.7 -y
conda activate ASAP
```

**2. Install nuscenes-devkit following the [official instructions](https://github.com/nutonomy/nuscenes-devkit).**
```shell
pip install nuscenes-devkit
```

**3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge nvidia
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**5. Install imageio for visualization (optional).**
```shell
conda install imageio
```
# Use the advanced method (object interpolation + temporal database) to generate 12Hz annotation.
1. Generate 20Hz LiDAR input pkl file for CenterPoint
```
bash scripts/nusc_20Hz_lidar_input_pkl.sh
```
2. Generate 20Hz detection results. We provide a template inference script for CenterPoint inference (**use MMDetection3D**):(use cpu for training ,changing in build_dp function)
```
python tools/test.py \
    $PATH_TO_ASAP/assets/centerpoint_20Hz_lidar_input.py \
    $pretrained_model_path (use the above)\
    --eval-options 'jsonfile_prefix=$PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/' \
    --out $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/rst.pkl \
    --format-only
```

3. Consequently, we obtain 20Hz inference results at $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/pts_bbox/results_nusc.json. Then we build the temporal database:(the .json may under ASAP/asset/)
```
bash scripts/nusc_20Hzlidar_instance-token_generator.sh --lidar_inf_rst_path $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/pts_bbox/results_nusc.json
```

4. Finaly, generate the 12Hz annotation:
```
bash scripts/ann_generator.sh 12 \
   --ann_strategy 'advanced' \
   --lidar_inf_rst_path ./out/lidar_20Hz/results_nusc_with_instance_token.json
```