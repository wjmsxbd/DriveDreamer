import torch
from torch import nn
from einops import rearrange,repeat
from ldm.util import instantiate_from_config

class IdentityWrapper(nn.Module):
    def __init__(self,unet_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)

    def forward(self,*args,**kwargs):
        pass


class StreamingSDWrapper(IdentityWrapper):
    def forward(self,
                x:torch.Tensor,
                t:torch.Tensor,
                c:dict,**kwargs):
        x = torch.cat((x,c.get("concat",torch.Tensor([]).type_as(x))),dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get('crossattn',None),
            y=c.get('vector',None),
            **kwargs
        )
    
    #FX TODO: clear feature cache
    def clear_model_cache(self,):
        # call self.diffusion_model.clear_model_cache()
        pass