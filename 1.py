import torch
from nuscenes.nuscenes import NuScenes
import pickle

# data_path = '/remote-home/share/nuScenes/nuScenes_advanced_infos_train.pkl'
# with open(data_path,'rb') as file:
#     data_infos = pickle.load(file)

# print(data_infos['infos'][0])
# token = data_infos['infos'][0]['token']
# nusc = NuScenes(version='advanced_12Hz_trainval',dataroot='/remote-home/share/nuScenes',verbose=True)
# x=1

x = torch.randn((1,10,5))
y = torch.randn((1,10,30))
linear1 = torch.nn.Linear(5,10)
linear2 = torch.nn.Linear(10,20)
linear3 = torch.nn.Linear(20,30)
x = linear1(x)
x = linear2(x)
x = linear3(x)
loss = torch.sum(y-x)
for param in linear2.parameters():
    param.requires_grad=False

loss.backward()
param = []
param.append(linear1.parameters())
param.append(linear3.parameters())
optimizer = torch.optim.Adam(linear1.parameters())
print(linear1.parameters())
optimizer.step()
print(linear1.parameters())