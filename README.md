# DriveDreamer
## 纯视觉训练

### Switch Branch

`
git checkout master
`

### Create conda env
`
conda create -n drivedreamer python=3.8
`

### Install Pytorch
`
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
`

### Installation
`
pip install -r requirement.txt
`
### Preparation

#### Download nuScenes-map-expansion-v1.3.zip
在[Nuscenes](https://www.nuscenes.org/download)官网上下载nuScenes-map-expansion-v1.3.zip
解压到当前路径，得到如下路径：
```
maps
├── basemap
├── expansion
├── prediction
```

#### Data Preparation
由于标注数据的频率只有2Hz，而相机捕获频率是12Hz，因此需要对非关键帧数据进行标注，得到[advanced_12Hz_trainval](https://drive.google.com/file/d/1t0kMU7Wk4CsH3f3rv-Utyin6PYtI0_Ns/view?usp=sharing)。然后在此基础上进行数据预处理，得到[nuScenes_advanced_infos_train.pkl](https://drive.google.com/file/d/1ulaKcqsu9p5z6w-2EoQyL42D95w0qOqY/view?usp=sharing)和[nuScenes_advanced_infos_val.pkl](https://drive.google.com/file/d/1K0YVPdk3OVDXMDG4agOGqi3Gv19OKspC/view?usp=sharing)

如果需要自己在本地处理可以参考12Hz_data_process.md文件

下载完数据之后需要按照以下路径存放数据
```
nuScenes
├── advanced_12Hz_trainval
├── nuScenes_advanced_infos_train.pkl
├── nuScenes_advanced_infos_val.pkl
```

nuScenes_advanced_infos_train和nuScenes_advanced_infos_val数据格式介绍如下：
```
nuScenes_advanced_infos_train : {
    'infos' : [
        ['token' : str
        'cams' : {
            'CAM_FRONT' : {
                'data_path' : str
                'type' : 'CAM_FRONT'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
            'CAM_FRONT_RIGHT' : {
                'data_path' : str
                'type' : 'CAM_FRONT_RIGHT'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
            'CAM_FRONT_LEFT' : {
                'data_path' : str
                'type' : 'CAM_FRONT_LEFT'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
            'CAM_BACK' : {
                'data_path' : str
                'type' : 'CAM_BACK'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
            'CAM_BACK_LEFT' : {
                'data_path' : str
                'type' : 'CAM_BACK_LEFT'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
            'CAM_BACK_RIGHT' : {
                'data_path' : str
                'type' : 'CAM_BACK_RIGHT'
                'sample_data_token' : str
                'timestamp' : int
                'cam_intrinsic' : numpy.ndarray
            }
        }
        'timestamp' : int
        ]
        ...
    ]
    'metadata' : {
        'version' : 'advanced_12Hz_trainval'
    }
}
```

#### Create stable_diffusion directory
`
mkdir stable_diffusion
`
#### Download Stable Diffusion v1.4 in stable_diffusion
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt)
#### Download AutoEncoderKL in stable_diffusion
[AutoEncoderKL](https://drive.google.com/file/d/1W-AMf_KlakUuLNbU9iCd-yKHeY01Pu23/view?usp=drive_link)

[kl-f8](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)

下载完的模型需要按照以下路径存放：
```
stable_diffusion
├── kl-f8.ckpt
├── sd-v1-4.ckpt
├── epoch=000001.ckpt
```

#### 

### Training
`
python main.py --base configs/first_stage_step1_config_mini.yaml --train True
`

## 视觉+Range Image

### Switch Branch

`
git checkout with_lidar
`


### Installation
`
pip install -r requirement.txt
`
### Preparation

#### Data Preparation
由于标注数据的频率只有2Hz，而相机捕获频率是12Hz，因此需要对非关键帧数据进行标注，得到[advanced_12Hz_trainval](https://drive.google.com/file/d/1t0kMU7Wk4CsH3f3rv-Utyin6PYtI0_Ns/view?usp=sharing)。然后在此基础上进行数据预处理，得到[nuScenes_advanced_infos_train.pkl](https://drive.google.com/file/d/1ulaKcqsu9p5z6w-2EoQyL42D95w0qOqY/view?usp=sharing)和[nuScenes_advanced_infos_val.pkl](https://drive.google.com/file/d/1K0YVPdk3OVDXMDG4agOGqi3Gv19OKspC/view?usp=sharing)

下载完数据之后需要按照以下路径存放数据
```
nuScenes
├── advanced_12Hz_trainval
├── nuScenes_advanced_infos_train.pkl
├── nuScenes_advanced_infos_val.pkl
```
#### Create stable_diffusion directory
`
mkdir stable_diffusion
`
#### Download Stable Diffusion v1.4 in stable_diffusion
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt)
#### Download AutoEncoderKL in stable_diffusion
[AutoEncoderKL](https://drive.google.com/file/d/1W-AMf_KlakUuLNbU9iCd-yKHeY01Pu23/view?usp=drive_link)

[kl-f8](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)

下载完的模型需要按照以下路径存放：
```
stable_diffusion
├── kl-f8.ckpt
├── sd-v1-4.ckpt
├── epoch=000001.ckpt
```

#### create hdmap data
修改configs/hdmap_generation.yaml里的数据路径
```
python data/hdmap_generation.py
```

#### 
#### clone ip_basic
在当前目录中clone ip_basic
```
git clone git@github.com:kujason/ip_basic.git
cd ip_basic
pip3 install -r requirements.txt
```
### Training

#### training lidar model
修改configs/autoencoder_lidar.yaml中的数据集的version和path
`
python main.py --base configs/autoencoder_lidar.yaml --train True
`
#### training diffusion model
修改global_condition.yaml中的数据集的version和path
其中如果要加载纯视觉训练的模型需要设置init_from_video_model=True且修改unet_config.params.ckpt_path，保留ignore_keys和modify_keys
如果要加载纯视觉训练+range image的模型需要设置init_from_video_model=False且更改unet_config.params.ckpt_path和ignore_keys
`
python main.py --base configs/global_condition.yaml --train True
`
