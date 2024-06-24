# DriveDreamer
## Installation
`
pip install -r requirement.txt
`

## Preparation
### Data Preparation
#### Training Data
训练数据来自NuScenes，但是由于标注数据的频率和摄像头捕捉图像的频率不一致，因此对非关键帧也进行了标注,最后得到[Advanced_12Hz_trainval](https://drive.google.com/file/d/1t0kMU7Wk4CsH3f3rv-Utyin6PYtI0_Ns/view?usp=drive_link)。

在训练前，对数据做了一部分预处理，整理后得到[nuScenes_advanced_infos_train.pkl](https://drive.google.com/file/d/1ulaKcqsu9p5z6w-2EoQyL42D95w0qOqY/view?usp=drive_link)和[nuScenes_advanced_infos_val.pkl](https://drive.google.com/file/d/1K0YVPdk3OVDXMDG4agOGqi3Gv19OKspC/view?usp=drive_link)。

它们包含了摄像头捕获的六个视角的图像，分别为CAM_FRONT,CAM_FRONT_RIGHT,CAM_FRONT_LEFT,CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT。数据格式如下：
```
{
    'infos':
    [
        {
            'token' : str
            'cams' : 
            {
                'CAM_FRONT':
                {
                    'data_path' : str
                    'type' : 'CAM_FRONT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }
                'CAM_FRONT_LEFT':
                {
                    'data_path' : str
                    'type' : 'CAM_FRONT_LEFT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }
                'CAM_FRONT_RIGHT':
                {
                    'data_path' : str
                    'type' : 'CAM_FRONT_RIGHT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }

                'CAM_BACK':
                {
                    'data_path' : str
                    'type' : 'CAM_FRONT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }
                'CAM_BACK_FRONT':
                {
                    'data_path' : str
                    'type' : 'CAM_BACK_FRONT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }'CAM_BACK_RIGHT':
                {
                    'data_path' : str
                    'type' : 'CAM_BACK_RIGHT'
                    'sample_data_token': str
                    'timestamp': int
                    'cam_intrinsic': numpy.ndarray
                }
            }
            'timestamp' : int
        }
        ...

    ]
    'metadata':
    {
        'version':'advanced_12Hz_trainval'
    }
}

```
#### Condition
Condition部分的数据通过在线处理得到。
##### HDMap
记录了前视图中['lane_divider','lane','ped_crossing']信息，大小为(H,W,3)
#### 3D Boxes
记录了在前视图中标注信息的8个顶点的位置坐标及标签，位置坐标大小为($N_B$,16)，标签经过CLIP之后大小为($N_B$,768)
#### Text
记录了该场景下的description，经过CLIP之后大小为(1,768)

#### DownLoad

### Create stable_diffusion directory
`
mkdir stable_diffusion
`
### Download Stable Diffusion v1.4 in stable_diffusion
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt)
### Download AutoEncoderKL in stable_diffusion
[AutoEncoderKL](https://drive.google.com/file/d/1W-AMf_KlakUuLNbU9iCd-yKHeY01Pu23/view?usp=drive_link)

### 

## Training
`
python main.py --base configs/first_stage_step1_config_mini.yaml --train True
`
