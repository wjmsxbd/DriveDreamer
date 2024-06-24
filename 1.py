import psutil
import os
import time
import torch
import pickle
# process = psutil.Process(42776)

# while True:
#     # 获取内存信息
#     mem_info = process.memory_info()
#     print(f"RSS: {mem_info.rss / 1024 / 1024:.2f} MB; VMS: {mem_info.vms / 1024 / 1024:.2f} MB")
#     time.sleep(10)  # 每10秒检查一次

model_path = 'stable_diffusion/vista.safetensors'
with open(model_path,'rb') as file:
    data = pickle.load(file)
print(data)