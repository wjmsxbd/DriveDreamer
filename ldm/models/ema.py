import torch
import torch.nn as nn

#TODO:finish Exponential Moving Average

class LitEma(nn.Module):
    def __init__(self,model,decay=0.9999,use_num_updates=True):
        pass