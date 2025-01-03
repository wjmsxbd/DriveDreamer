import torch
import torch.nn as nn

class PostProcess(nn.Module):
    def __init__(self,cfg):
        super(PostProcess,self).__init__()
        self.cfg = cfg

    def forward(self,output,):
        center_label,offset_label = output['center_label_pred'],output['offset_label_pred']
        
