# DriveDreamer
## 纯视觉训练
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

#### 

### Training

#### 在训练的时候对图片进行分块

`
python main.py --base configs/first_stage_step1_config_split_on_train.yaml --train True
`
#### 在预处理的时候对图片进行分块
`
python main.py --base configs/first_stage_step1_config_split_pre_train.yaml --train True
`
