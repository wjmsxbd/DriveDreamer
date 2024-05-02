import glob

import os

data_path = './nuScenes'
datasets = glob.glob(data_path+"/*")
for data in datasets:
    os.rename(data,data[:-4])