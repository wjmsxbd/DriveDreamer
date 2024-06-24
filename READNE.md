# DriveDreamer
## Installation
`
pip install -r requirement.txt
`

## Preparation
### Download Stable Diffusion v1.4
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt)
### Download AutoEncoderKL
[AutoEncoderKL](https://drive.google.com/file/d/1W-AMf_KlakUuLNbU9iCd-yKHeY01Pu23/view?usp=drive_link)

### 

## Training
`
python main.py --base configs/first_stage_step1_config_mini.yaml --train True
`
