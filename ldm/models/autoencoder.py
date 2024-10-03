import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder,Encoder_Diffusion,Decoder_Diffusion,Decoder_Lidar,Decoder_Temporal,Encoder_Temporal
from ldm.models.ema import LitEma
# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder,Decoder
from ldm.modules.distributions.distributions import DiagonalGuassianDistribution
from packaging import version
from ldm.util import instantiate_from_config
import math
import clip
from einops import repeat,rearrange
from torch.optim.lr_scheduler import LambdaLR
from scipy.special import kl_div
import torchvision
from pytorch_fid import fid_score
torch.set_default_dtype(torch.float32)

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="reference_image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Diffusion(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec



class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="reference_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 train_downsample=False,
                 trainable=False,):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.train_downsample=train_downsample
        self.trainable = trainable
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_finetune = use_finetune

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self,x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        return posterior
    
    def decode(self,z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec,posterior

    def get_losses(self,batch,batch_idx,optimizer_idx):
        reconstructions,posterior = self(batch)
        if optimizer_idx == 0:
            aeloss,log_dict_ae = self.loss(batch,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            return aeloss,log_dict_ae
        else:
            discloss, log_dict_disc = self.loss(batch, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            return discloss,log_dict_disc
            

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[...,None]
        x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)

        if optimizer_idx == 0:
            #train encoder + decoder + logvar
            aeloss,log_dict_ae = self.loss(inputs,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.train_downsample:
            for name,param in self.named_parameters():
                if 'downsample' in name:
                    params.append(param)
                    print(f"add:{name}")
            opt_ae = torch.optim.Adam(params)
        else:
            print("add all!!!!!!!!!!!!!!")
            opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr,betas=(0.5,0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,betas=(0.5,0.9))
        return [opt_ae,opt_disc],[]
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self,batch,only_inputs=False,**kwargs):
        log = dict()
        x = self.get_input(batch,self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec,posterior = self(x)
            if x.shape[1] > 3:
                #colorize with random projection
                assert xrec.shape[1]>3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    def to_rgb(self,x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

def Normalize(in_channels,num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups,num_channels=in_channels,eps=1e-6,affine=True)

def nonlinearity(x):
    #swish 
    return x * torch.sigmoid(x)

class Fourier_Embedding(nn.Module):
    def __init__(self,in_channels,ch,context_dims,out_channels=None,sigma: float = 1,trainable: bool = False,device=None,dtype=torch.float32):
        """
        For input feature vector `x`, apply linear transformation
        followed by concatenated sine and cosine: `[sin(Ax), cos(Ax)]`.

        - Input: `(*, in_features)`, where the last dimension is the feature dimension
        - Output: `(*, out_features)`, where all but the last dimension are the same shape as the input

        Unlike element-wise sinusoidal encoding,
        Fourier features capture the interaction of high-dimensional input features.

        Note that a bias term is not necessary in the linear transformation,
        because `sin(wx+b)` and `cos(wx+b)` can both be represented by
        linear combinations of `sin(wx)` and `cos(wx)`, which can be learned in the
        subsequent linear layer.
        (https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

        In learnable Fourier features (https://arxiv.org/abs/2106.02795),
        the linear transformation is trainable after random normal initialization.
        They also use an MLP (linear, GELU, linear) following this Fourier feature layer,
        because the goal is to produce a positional encoding added to word embedding.
        We do not implement the MLP here, since a user should be free to decide
        which layers to follow; this keeps the modularity of FourierFeatures.

        Also note that in learnable Fourier features, the final output
        after sine/cosine is divided by sqrt(H_out). We do not implement that yet.

        Parameters
        ----------
        in_features: int
            Size of each input sample

        out_features: int
            Size of each output sample

        sigma: float
            Standard deviation of 0-centered normal distribution,
            which is used to initialize linear transformation parameters.

        trainable: bool, default: False
            If True, the linear transformation is trainable by gradient descent.
        """
        super().__init__()
        assert ch[0] % 2 == 0,"number of out_features must be even"
        self.in_channels = in_channels
        self.ch = ch
        self.sigma = sigma
        self.trainable = bool(trainable)
        if out_channels is None:
            out_channels = ch[-1]
        #define linear layer
        self.linear = nn.Linear(
            in_features=in_channels,out_features=ch[0] // 2,
            bias=False,device=device,dtype=dtype
        )

        nn.init.normal_(self.linear.weight.data,mean=0.0,std=self.sigma)

        self.linear2 = nn.Linear(in_features=ch[0]+context_dims,out_features=ch[1],bias=True,device=device,dtype=dtype)
        self.linear3 = nn.Linear(in_features=ch[1]+context_dims,out_features=ch[2],bias=True,device=device,dtype=dtype)
        self.linear4 = nn.Linear(in_features=ch[2]+context_dims,out_features=out_channels,bias=True,device=device,dtype=dtype)

        self.norm1 = torch.nn.BatchNorm1d(ch[0])
        self.norm2 = torch.nn.BatchNorm1d(ch[1])
        self.norm3 = torch.nn.BatchNorm1d(ch[2])

        models = [self.linear,self.linear2,self.linear3]        
        #freeze layer if not trainable
        for model in models:
            for param in model.parameters():
                param.requires_grad_(self.trainable)
        
    def encode(self,x,category):
        return self(x,category)

    def forward(self,x:torch.Tensor,category:torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = math.tau * x
        x = torch.cat((x.sin(),x.cos()),dim=-1)
        x = rearrange(x,'b n c -> b c n')
        x = self.norm1(x)
        x = nonlinearity(x)
        x = rearrange(x,'b c n -> b n c')
        x = torch.cat((x,category),dim=-1)
        x = self.linear2(x)
        x = rearrange(x,'b n c -> b c n')
        x = self.norm2(x)
        x = nonlinearity(x)
        x = rearrange(x,'b c n -> b n c')
        x = torch.cat((x,category),dim=-1)
        x = self.linear3(x)
        x = rearrange(x,'b n c -> b c n')
        x = self.norm3(x)
        x = nonlinearity(x)
        x = rearrange(x,'b c n -> b n c')
        x = torch.cat((x,category),dim=-1)
        x = self.linear4(x)
        return x

        
class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self,version='ViT-L/14',device='cuda',max_length=77,n_repeat=1,normalize=True):
        super().__init__()
        self.model,_ = clip.load(version,jit=False,device='cpu')
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize
        

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self,text):
        tokens = clip.tokenize(text).cuda()
        # tokens = clip.tokenize(text)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z,dim=1,keepdim=True)
        return z
    
    def encode(self,text):
        z = self(text)
        if z.ndim == 2:
            z = z[:,None,:]
        z = repeat(z,'b 1 d -> b k d',k = self.n_repeat)
        return z
    

class AutoencoderKL_Diffusion(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="reference_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 movie_len=None,
                 trainable=False,):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Diffusion(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.movie_len = movie_len
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.use_finetune=use_finetune
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        # for name,param in self.named_parameters():
        #     flag = False
        #     for ik in ignore_keys:
        #         if name.startswith(ik):
        #             flag=True
        #     if flag or name.startswith('decoder'):
        #         print(f"add {name} grad = true")
        #         param.requires_grad=True
        #     else:
        #         param.requires_grad=False
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
                    
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self,x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        return posterior
    
    def decode(self,z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec,posterior

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 5:
            x = rearrange(x,'b n h w c -> (b n) c h w').contiguous()
        else:
            if len(x.shape) == 3:
                x = x[...,None]
            x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)

        if optimizer_idx == 0:
            #train encoder + decoder + logvar
            aeloss,log_dict_ae = self.loss(inputs,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.use_finetune:
            opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr,betas=(0.5,0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr,betas=(0.5,0.9))
            return [opt_ae,opt_disc],[]
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr,betas=(0.5,0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr,betas=(0.5,0.9))
            return [opt_ae,opt_disc],[]
    
    def get_parameters(self):
        return list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def get_losses(self,batch,batch_idx,optimizer_idx,target=None):
        reconstructions,posterior = self(batch)
        if optimizer_idx == 0:
            if target is None:
                aeloss,log_dict_ae = self.loss(batch,reconstructions,posterior,optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            else:
                aeloss,log_dict_ae = self.loss(target,reconstructions,posterior,optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss,log_dict_ae
        else:
            if target is None:
                discloss, log_dict_disc = self.loss(batch, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
            else:
                discloss, log_dict_disc = self.loss(target, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            return discloss,log_dict_disc
    
    @torch.no_grad()
    def log_video(self,batch,N=8,n_row=4,split=None):
        log = dict()
        x = batch['reference_image']
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        x = rearrange(x,'b n h w c -> b n c h w')
        log['inputs'] = x
        x = self.get_input(batch,self.image_key)
        dec,_ = self(x)
        log['reconstruction'] = dec.reshape((N,-1)+dec.shape[1:])
        return log



    # @torch.no_grad()
    # def log_images(self,batch,only_inputs=False,**kwargs):
    #     log = dict()
    #     x = self.get_input(batch,self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec,posterior = self(x)
    #         if x.shape[1] > 3:
    #             #colorize with random projection
    #             assert xrec.shape[1]>3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log
    # def to_rgb(self,x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    #     return x

class AutoencoderKL_Temporal(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="reference_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 movie_len=None,
                 trainable=False,
                 use_ema=False,
                 safetensor_path=None,
                 load_from_AECombine=False):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Temporal(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.movie_len = movie_len
        self.use_ema = use_ema
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.use_finetune=use_finetune
        if load_from_AECombine:
            self.load_from_AECombine(ckpt_path)
        else:
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            if safetensor_path is not None:
                self.init_from_safetensor(safetensor_path)
        
    def load_from_AECombine(self,path):
        sd = torch.load(path,map_location='cpu')["state_dict"]
        keys = list(sd.keys())
        weight = dict()
        for k in keys:
            if 'image_model' in k:
                weight[k[len('image_model')+1:]] = sd[k].clone()
                print("Add key {} from sd".format(k))
        self.load_state_dict(weight,strict=False)
        print(f"Resotred from {path}")

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        # for name,param in self.named_parameters():
        #     flag = False
        #     for ik in ignore_keys:
        #         if name.startswith(ik):
        #             flag=True
        #     if flag or name.startswith('decoder'):
        #         print(f"add {name} grad = true")
        #         param.requires_grad=True
        #     else:
        #         param.requires_grad=False
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
                    
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def init_from_safetensor(self,safetensor_path):
        from safetensors import safe_open
        sd = dict()
        with safe_open(safetensor_path,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key.startswith('first_stage_model.decoder'):
                    sd[key[len("first_stage_model")+1:]] = f.get_tensor(key)
                if key.startswith('conditioner.embedders.3.encoder.encoder'):
                    if key.startswith('conditioner.embedders.3.encoder.encoder.quant_conv'):
                        sd[key[len('conditioner.embedders.3.encoder.encoder')+1:]] =  f.get_tensor(key)
                    else:
                        sd[key[len('conditioner.embedders.3.encoder')+1:]] = f.get_tensor(key)

        missing,unexpected = self.load_state_dict(sd,strict=False)
        print(f"missing:{missing},unexpected:{unexpected}")

    def encode(self,x,return_temp_output=False):
        h = self.encoder(x,return_temp_output=return_temp_output)
        if return_temp_output:
            h,temp = h
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        if return_temp_output:
            return posterior,temp
        return posterior
    
    def decode(self,z,return_temp_output=False):
        # z = self.post_quant_conv(z)
        if return_temp_output:
            dec,_ = self.decoder(z,return_temp_output)
            return dec,_
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True,return_temp_output=False):
        posterior = self.encode(input,return_temp_output=return_temp_output)
        if return_temp_output:
            posterior,enc_temp = posterior
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z,return_temp_output=return_temp_output)
        if return_temp_output:
            dec,dec_temp = dec
            return dec,posterior,enc_temp,dec_temp
        return dec,posterior

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 5:
            x = rearrange(x,'b n h w c -> (b n) c h w').contiguous()
        else:
            if len(x.shape) == 3:
                x = x[...,None]
            x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)

        if optimizer_idx == 0:
            #train encoder + decoder + logvar
            aeloss,log_dict_ae = self.loss(inputs,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.use_finetune:
            opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr,betas=(0.5,0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr,betas=(0.5,0.9))
            return [opt_ae,opt_disc],[]
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr,betas=(0.5,0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr,betas=(0.5,0.9))
            return [opt_ae,opt_disc],[]
    
    def get_parameters(self):
        return list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def get_losses(self,batch,batch_idx,optimizer_idx,target=None):
        reconstructions,posterior = self(batch)
        if optimizer_idx == 0:
            if target is None:
                aeloss,log_dict_ae = self.loss(batch,reconstructions,posterior,optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            else:
                aeloss,log_dict_ae = self.loss(target,reconstructions,posterior,optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss,log_dict_ae
        else:
            if target is None:
                discloss, log_dict_disc = self.loss(batch, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
            else:
                discloss, log_dict_disc = self.loss(target, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            return discloss,log_dict_disc
    
    @torch.no_grad()
    def log_video(self,batch,N=8,n_row=4,split=None):
        log = dict()
        x = batch['reference_image']
        N = min(x.shape[0],N)
        n_row = min(x.shape[0],n_row)
        x = rearrange(x,'b n h w c -> b n c h w')
        log['inputs'] = x
        x = self.get_input(batch,self.image_key)
        dec,_ = self(x)
        log['reconstruction'] = dec.reshape((N,-1)+dec.shape[1:])
        return log



    # @torch.no_grad()
    # def log_images(self,batch,only_inputs=False,**kwargs):
    #     log = dict()
    #     x = self.get_input(batch,self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec,posterior = self(x)
    #         if x.shape[1] > 3:
    #             #colorize with random projection
    #             assert xrec.shape[1]>3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log
    # def to_rgb(self,x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    #     return x

class Autoencoder_Lidar(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="range_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 train_downsample=False,
                 trainable=False,):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Diffusion(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.train_downsample=train_downsample
        self.trainable = trainable
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_finetune = use_finetune

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self,x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        return posterior
    
    def decode(self,z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec,posterior
        

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[...,None]
        x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        loss,log_dict = self.loss(inputs,reconstructions,posterior,split="train")
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict,prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        loss,log_dict = self.loss(inputs,reconstructions,posterior,split="val")
        self.log("val/loss_simple", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict,prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                list(self.decoder.parameters())+
                                list(self.quant_conv.parameters())+
                                list(self.post_quant_conv.parameters()),
                                lr=lr,betas=(0.5,0.9))
        return opt_ae
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self,batch,only_inputs=False,**kwargs):
        log = dict()
        x = self.get_input(batch,self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec,posterior = self(x)
            if x.shape[1] > 3:
                #colorize with random projection
                assert xrec.shape[1]>3
                x = self.to_rgb(x)
                # xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))[:,2]
            log["samples"] = repeat(log["samples"],'b h w -> b c h w',c=3)
            label = xrec[:,:2]
            label = torch.argmax(label,dim=1)
            mask = (label == 0)
            range_image = xrec[:,2]
            range_image[mask] = -1
            xrec = repeat(range_image,'b h w -> b c h w',c=3)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    def to_rgb(self,x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class Autoencoder_Lidar2(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="range_image",
                 gt_image_key="range_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 train_downsample=False,
                 trainable=False,
                 safetensor_path=None,
                 img_size=[128,256],
                 load_from_AECombine=False,):
        super().__init__()
        self.image_key = image_key
        self.gt_image_key = gt_image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Diffusion(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.train_downsample=train_downsample
        self.trainable = trainable

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if load_from_AECombine:
            self.load_from_AECombine(ckpt_path)
        else:
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_finetune = use_finetune
        self.img_size = img_size
        self.theta_up = torch.pi / 12
        self.theta_down = -torch.pi / 6
        self.theta_res = (self.theta_up - self.theta_down) / self.img_size[0] / 2
        self.phi_res = (torch.pi / 3) / self.img_size[1] / 2

    def load_from_AECombine(self,path):
        sd = torch.load(path,map_location='cpu')["state_dict"]
        keys = list(sd.keys())
        weight = dict()
        for k in keys:
            if 'lidar_model' in k:
                weight[k[len('lidar_model')+1:]] = sd[k].clone()
                print("Add key {} from sd".format(k))
        self.load_state_dict(weight,strict=False)
        print(f"Resotred from {path}")


    def calc_3D_point_loss(self,inputs,reconstructions):
        reconstructions = (torch.clip(reconstructions,-1,1) + 1.) / 2.
        mask = reconstructions >= 0.05
        reconstructions = reconstructions * mask
        reconstructions = (reconstructions * 255.)
        inputs = (torch.clip(inputs,-1,1) + 1.) / 2.
        inputs = (inputs * 255.)
        b,c,h,w = inputs.shape
        x_coords,y_coords = torch.meshgrid(torch.arange(0,h),torch.arange(0,w))
        inputs[:,0] = x_coords
        inputs[:,1] = y_coords
        reconstructions[:,0] = x_coords
        reconstructions[:,1] = y_coords
        inputs_phi = ((1+inputs[:,0:1]) / self.img_size[1] - 1) * torch.pi / 6 + self.phi_res
        reconstructions_phi = ((1+reconstructions[:,0:1]) / self.img_size[1] - 1) * torch.pi / 6 + self.phi_res
        inputs_theta = (self.theta_up) - (self.theta_up - self.theta_down) * ((inputs[:,1:2] + 1) - 1./2) / self.img_size[0]  + self.theta_res
        reconstructions_theta = (self.theta_up) - (self.theta_up - self.theta_down) * ((reconstructions[:,1:2] + 1) - 1./2) / self.img_size[0] + self.theta_res
        inputs_r = inputs[:,2:3]
        reconstructions_r = reconstructions[:,2:3]
        point_x_inputs = inputs_r * torch.cos(inputs_theta) * torch.sin(inputs_phi)
        point_y_inputs = inputs_r * torch.cos(inputs_theta) * torch.cos(inputs_phi)
        point_z_inputs = inputs_r * torch.sin(inputs_theta)
        point_x_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.sin(reconstructions_phi)
        point_y_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.cos(reconstructions_phi)
        point_z_rec = reconstructions_r * torch.sin(reconstructions_theta)
        points_inputs = torch.cat([point_x_inputs,point_y_inputs,point_z_inputs],dim=1)
        points_rec = torch.cat([point_x_rec,point_y_rec,point_z_rec],dim=1)
        point_loss = torch.sum((points_inputs - points_rec) ** 2)
        point_loss = point_loss / b
        return point_loss

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self,x,return_temp_output=False):
        h = self.encoder(x,return_temp_output=return_temp_output)
        if return_temp_output:
            h,temp = h
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        if return_temp_output:
            return posterior,temp
        return posterior
    
    def decode(self,z,return_temp_output=False):
        z = self.post_quant_conv(z)
        if return_temp_output == True:
            dec,_ = self.decoder(z,return_temp_output)
            return dec,_
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True,return_temp_output=False):
        posterior = self.encode(input,return_temp_output=return_temp_output)
        if return_temp_output:
            posterior,enc_temp = posterior
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z,return_temp_output=return_temp_output)
        if return_temp_output:
            dec,dec_temp = dec
            return dec,posterior,enc_temp,dec_temp
        return dec,posterior

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 5:
            x = rearrange(x,'b n h w c -> (b n) c h w').contiguous()
        else:
            if len(x.shape) == 3:
                x = x[...,None]
            x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        inputs = self.get_input(batch,self.gt_image_key)
        if optimizer_idx == 0:
            #train encoder + decoder + logvar
            aeloss,log_dict_ae = self.loss(inputs,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        inputs = self.get_input(batch,self.gt_image_key)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                list(self.decoder.parameters())+
                                list(self.quant_conv.parameters())+
                                list(self.post_quant_conv.parameters()),
                                lr=lr,betas=(0.5,0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,betas=(0.5,0.9))
        return [opt_ae,opt_disc],[]
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self,batch,only_inputs=False,**kwargs):
        log = dict()
        x = self.get_input(batch,self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec,posterior = self(x)
            if x.shape[1] > 3:
                #colorize with random projection
                assert xrec.shape[1]>3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    def to_rgb(self,x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class Autoencoder_Lidar3(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="range_image",
                 gt_image_key="range_image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_finetune=False,
                 train_downsample=False,
                 trainable=False,
                 safetensor_path=None,
                 img_size=None):
        super().__init__()
        self.image_key = image_key
        self.gt_image_key = gt_image_key
        self.encoder = Encoder_Diffusion(**ddconfig)
        self.decoder = Decoder_Temporal(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.train_downsample=train_downsample
        self.trainable = trainable
        self.img_size = img_size
        self.theta_up = torch.pi / 12
        self.theta_down = -torch.pi / 6
        self.theta_res = (self.theta_up - self.theta_down) / self.img_size[0] / 2
        self.phi_res = (torch.pi / 3) / self.img_size[1] / 2

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_finetune = use_finetune
        if safetensor_path is not None:
            self.init_from_safetensor(safetensor_path)

    def calc_3D_point_loss(self,inputs,reconstructions):
        reconstructions = (torch.clip(reconstructions,-1,1) + 1.) / 2.
        mask = reconstructions >= 0.05
        reconstructions = reconstructions * mask
        reconstructions = (reconstructions * 255.)
        inputs = (torch.clip(inputs,-1,1) + 1.) / 2.
        inputs = (inputs * 255.)
        b,c,h,w = inputs.shape
        x_coords,y_coords = torch.meshgrid(torch.arange(0,h),torch.arange(0,w))
        inputs[:,0] = x_coords
        inputs[:,1] = y_coords
        reconstructions[:,0] = x_coords
        reconstructions[:,1] = y_coords
        inputs_phi = ((1+inputs[:,0:1]) / self.img_size[1] - 1) * torch.pi / 6 + self.phi_res
        reconstructions_phi = ((1+reconstructions[:,0:1]) / self.img_size[1] - 1) * torch.pi / 6 + self.phi_res
        inputs_theta = (self.theta_up) - (self.theta_up - self.theta_down) * ((inputs[:,1:2] + 1) - 1./2) / self.img_size[0]  + self.theta_res
        reconstructions_theta = (self.theta_up) - (self.theta_up - self.theta_down) * ((reconstructions[:,1:2] + 1) - 1./2) / self.img_size[0] + self.theta_res
        inputs_r = inputs[:,2:3]
        reconstructions_r = reconstructions[:,2:3]
        point_x_inputs = inputs_r * torch.cos(inputs_theta) * torch.sin(inputs_phi)
        point_y_inputs = inputs_r * torch.cos(inputs_theta) * torch.cos(inputs_phi)
        point_z_inputs = inputs_r * torch.sin(inputs_theta)
        point_x_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.sin(reconstructions_phi)
        point_y_rec = reconstructions_r * torch.cos(reconstructions_theta) * torch.cos(reconstructions_phi)
        point_z_rec = reconstructions_r * torch.sin(reconstructions_theta)
        points_inputs = torch.cat([point_x_inputs,point_y_inputs,point_z_inputs],dim=1)
        points_rec = torch.cat([point_x_rec,point_y_rec,point_z_rec],dim=1)
        point_loss = torch.sum((points_inputs - points_rec) ** 2)
        point_loss = point_loss / b
        return point_loss

    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def init_from_safetensor(self,safetensor_path):
        from safetensors import safe_open
        sd = dict()
        with safe_open(safetensor_path,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key.startswith('first_stage_model.decoder'):
                    sd[key[len("first_stage_model")+1:]] = f.get_tensor(key)
                # if key.startswith('conditioner.embedders.3.encoder.encoder'):
                #     if key.startswith('conditioner.embedders.3.encoder.encoder.quant_conv'):
                #         sd[key[len('conditioner.embedders.3.encoder.encoder')+1:]] =  f.get_tensor(key)
                #     else:
                #         sd[key[len('conditioner.embedders.3.encoder')+1:]] = f.get_tensor(key)

        missing,unexpected = self.load_state_dict(sd,strict=False)
        print(f"missing:{missing},unexpected:{unexpected}")

    def encode(self,x,return_temp_output=False):
        h = self.encoder(x,return_temp_output=return_temp_output)
        if return_temp_output:
            h,temp = h
        moments = self.quant_conv(h)
        posterior = DiagonalGuassianDistribution(moments)
        if return_temp_output:
            return posterior,temp
        return posterior
    
    def decode(self,z,return_temp_output=False):
        # z = self.post_quant_conv(z)
        if return_temp_output == True:
            dec,_ = self.decoder(z,return_temp_output)
            return dec,_
        dec = self.decoder(z)
        return dec
    
    def forward(self,input,sample_posterior=True,return_temp_output=False):
        posterior = self.encode(input,return_temp_output=return_temp_output)
        if return_temp_output:
            posterior,enc_temp = posterior
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z,return_temp_output=return_temp_output)
        if return_temp_output:
            dec,dec_temp = dec
            return dec,posterior,enc_temp,dec_temp
        return dec,posterior

    #TODO:conver b h w c -> b c h w
    def get_input(self,batch,k):
        x = batch[k]
        if len(x.shape) == 5:
            x = rearrange(x,'b n h w c -> (b n) c h w').contiguous()
        else:
            if len(x.shape) == 3:
                x = x[...,None]
            x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self,batch,batch_idx,optimizer_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        inputs = self.get_input(batch,self.gt_image_key)
        if optimizer_idx == 0:
            #train encoder + decoder + logvar
            aeloss,log_dict_ae = self.loss(inputs,reconstructions,posterior,optimizer_idx, self.global_step,
                                           last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
    def validation_step(self,batch,batch_idx):
        inputs = self.get_input(batch,self.image_key)
        reconstructions,posterior = self(inputs)
        inputs = self.get_input(batch,self.gt_image_key)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                list(self.decoder.parameters())+
                                list(self.quant_conv.parameters())+
                                list(self.post_quant_conv.parameters()),
                                lr=lr,betas=(0.5,0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,betas=(0.5,0.9))
        return [opt_ae,opt_disc],[]
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self,batch,only_inputs=False,**kwargs):
        log = dict()
        x = self.get_input(batch,self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec,posterior = self(x)
            if x.shape[1] > 3:
                #colorize with random projection
                assert xrec.shape[1]>3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    def to_rgb(self,x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
class AE_Combine(pl.LightningModule):
    def __init__(self,
                 image_config,
                 lidar_config,
                 monitor=None,
                 use_similar='KL',
                 ckpt_path=None,
                 ignore_keys=[],
                 lidar_type='Decoder_Temporal'):
        super().__init__()
        self.image_model = instantiate_from_config(image_config)
        self.lidar_model = instantiate_from_config(lidar_config)
        self.use_similar = use_similar
        self.lidar_type = lidar_type
        if not monitor is None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode_first_stage(self,x,encoder,return_temp_output=False):
        return encoder.encode(x,return_temp_output=return_temp_output)
    
    def decode_first_stage(self,x,decoder,return_temp_output=False):
        return decoder.decode(x,return_temp_output=return_temp_output)
    
    def calc_single_similarity(self,cam_enc,lidar_enc):
        eps = 1e-7
        data = torch.tensor([0.],dtype=torch.float32,device=cam_enc[0].device)
        if self.use_similar == 'JS':
            for i in range(len(cam_enc)):
                min_value = cam_enc[i].min()
                max_value = cam_enc[i].max()
                cam_enc[i] = (cam_enc[i] - min_value) / (max_value - min_value)
                # cam_enc[i] = (cam_enc[i] * 255.).to(torch.uint8)
                _,_,h,w = cam_enc[i].shape
                l = h * w
                cam_enc[i] = rearrange(cam_enc[i],'b c h w -> (b h w) c')
                min_value = lidar_enc[i].min()
                max_value = lidar_enc[i].max()
                lidar_enc[i] = (lidar_enc[i] - min_value) / (max_value - min_value)
                # lidar_enc[i] = (lidar_enc[i] * 255.).to(torch.float32)
                lidar_enc[i] = rearrange(lidar_enc[i],'b c h w -> (b h w) c')
                JS = torch.tensor([0.],device=cam_enc[i].device)
                for j in range(cam_enc[i].shape[1]):
                    xx = torch.histc(cam_enc[i][:,j],bins=256)
                    lidar_xx = torch.histc(lidar_enc[i][:,j],bins=256)
                    lidar_xx = lidar_xx / l + eps
                    xx = xx / l + eps
                    m = (xx + lidar_xx) * 0.5
                    kl_pm = torch.sum((torch.kl_div(xx,m)))
                    kl_qm = torch.sum(torch.kl_div(lidar_xx,m))
                    js = 0.5 * (kl_pm + kl_qm)
                    JS += js
                data += JS 
            return data

    def calc_similarity(self,cam_enc,cam_dec,lidar_enc,lidar_dec):
        if self.use_similar == 'JS':
            gt = self.calc_single_similarity(cam_enc,lidar_enc)
            rec = self.calc_single_similarity(cam_dec,lidar_dec)
            similarity = (gt - rec).mean()
            return similarity
                
    def init_from_ckpt(self,path,ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def training_step(self,batch,batch_idx):
        
        cam_inputs = self.image_model.get_input(batch,self.image_model.image_key)        
        cam_reconstructions,cam_posterior,cam_enc_temp,cam_dec_temp = self.image_model(cam_inputs,return_temp_output=True)
        range_image = self.lidar_model.get_input(batch,self.lidar_model.image_key)
        lidar_reconstructions,lidar_posterior,lidar_enc_temp,lidar_dec_temp = self.lidar_model(range_image,return_temp_output=True)
        optimizer_idx = 0
        if optimizer_idx == 0:
            cam_aeloss,log_dict_cam_ae = self.image_model.loss(cam_inputs,cam_reconstructions,cam_posterior,optimizer_idx,self.global_step,last_layer=self.image_model.get_last_layer(), split="train")
            self.log("cam_aeloss", cam_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_cam_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            lidar_aeloss,log_dict_lidar_ae = self.lidar_model.loss(range_image,lidar_reconstructions,lidar_posterior,optimizer_idx,self.global_step,last_layer=self.lidar_model.get_last_layer(),split="train")
            self.log("lidar_aeloss", lidar_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_lidar_ae,prog_bar=False, logger=True, on_step=True, on_epoch=False)
            similarity = self.calc_similarity(cam_enc_temp,cam_dec_temp,lidar_enc_temp,lidar_dec_temp)
            point_loss = self.lidar_model.calc_3D_point_loss(range_image,lidar_reconstructions)
            loss = cam_aeloss / cam_aeloss.detach() + lidar_aeloss / lidar_aeloss.detach() + similarity / similarity.detach() + point_loss / point_loss.detach()
            self.log("similarity", similarity, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("point_loss", point_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss
        else:
            cam_discloss, log_dict_cam_disc = self.image_model.loss(cam_inputs, cam_reconstructions, cam_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.image_model.get_last_layer(), split="train")

            self.log("cam_discloss", cam_discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_cam_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            lidar_discloss, log_dict_lidar_disc = self.lidar_model.loss(range_image, lidar_reconstructions, lidar_posterior, optimizer_idx, self.global_step,
                                                last_layer=self.lidar_model.get_last_layer(), split="train")

            self.log("lidar_discloss", lidar_discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_lidar_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            similarity = self.calc_similarity(cam_enc_temp,cam_dec_temp,lidar_enc_temp,lidar_dec_temp)
            loss = cam_discloss / cam_discloss.detach() + lidar_discloss / lidar_discloss.detach() + similarity / similarity.detach()
            return loss


    def validation_step(self,batch,batch_idx):
        cam_inputs = self.image_model.get_input(batch,self.image_model.image_key)        
        cam_reconstructions,cam_posterior = self.image_model(cam_inputs)
        range_image = self.lidar_model.get_input(batch,self.lidar_model.image_key)
        lidar_reconstructions,lidar_posterior = self.lidar_model(range_image)
        cam_aeloss, log_dict_cam_ae = self.image_model.loss(cam_inputs, cam_reconstructions, cam_posterior, 0, self.global_step,
                                        last_layer=self.image_model.get_last_layer(), split="val")
        # cam_discloss, log_dict_cam_disc = self.image_model.loss(cam_inputs, cam_reconstructions, cam_posterior, 1, self.global_step,
        #                                     last_layer=self.image_model.get_last_layer(), split="val")
        lidar_aeloss, log_dict_lidar_ae = self.lidar_model.loss(range_image, lidar_reconstructions, lidar_posterior, 0, self.global_step,
                                        last_layer=self.lidar_model.get_last_layer(), split="val")
        # lidar_discloss, log_dict_lidar_disc = self.lidar_model.loss(range_image, lidar_reconstructions, lidar_posterior, 1, self.global_step,
        #                                     last_layer=self.lidar_model.get_last_layer(), split="val")
        point_loss = self.lidar_model.calc_3D_point_loss(range_image,lidar_reconstructions)
        rec_loss = log_dict_cam_ae['val/rec_loss'] + log_dict_lidar_ae['val/rec_loss'] + point_loss
        self.log("val/rec_loss", rec_loss)
        self.log("val/point_loss", point_loss)
        self.log_dict(log_dict_cam_ae)
        # self.log_dict(log_dict_cam_disc)
        self.log_dict(log_dict_lidar_ae)
        # self.log_dict(log_dict_lidar_disc)
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.lidar_type == 'Decoder_Diffusion':
            opt_ae = torch.optim.Adam(list(self.image_model.decoder.parameters())+
                                  list(self.lidar_model.decoder.parameters())+
                                  list(self.lidar_model.post_quant_conv.parameters()),
                                  lr=lr,betas=(0.5,0.9))
        else:
            opt_ae = torch.optim.Adam(list(self.image_model.decoder.parameters())+
                                    list(self.lidar_model.decoder.parameters()),
                                    lr=lr,betas=(0.5,0.9))
        # opt_disc = torch.optim.Adam(list(self.image_model.loss.discriminator.parameters())+
        #                             list(self.lidar_model.loss.discriminator.parameters()),
        #                             lr=lr,betas=(0.5,0.9))
        return [opt_ae],[]

    @torch.no_grad()
    def log_video(self,batch,N=8,n_row=4,split=None):
        log = dict()
        x = batch['reference_image']
        b = x.shape[0]
        cam_inputs = self.image_model.get_input(batch,self.image_model.image_key)        
        cam_reconstructions,cam_posterior = self.image_model(cam_inputs)
        range_image = self.lidar_model.get_input(batch,self.lidar_model.image_key)
        lidar_reconstructions,lidar_posterior = self.lidar_model(range_image)
        print(f"point_loss:{self.lidar_model.calc_3D_point_loss(range_image,lidar_reconstructions)}")
        # x = batch['reference_image']
        # N = min(x.shape[0],N)
        # n_row = min(x.shape[0],n_row)
        # x = rearrange(x,'b n h w c -> b n c h w')
        cam_inputs = rearrange(cam_inputs,'(b n) c h w ->  b n c h w',b=b)
        range_image = rearrange(range_image,'(b n) c h w ->  b n c h w',b=b)
        log['inputs'] = cam_inputs
        log['lidar_inputs'] = range_image
        cam_reconstructions = rearrange(cam_reconstructions,'(b n) c h w -> b n c h w',b=b)
        lidar_reconstructions = rearrange(lidar_reconstructions,'(b n) c h w -> b n c h w',b=b)
        log['reconstruction'] = cam_reconstructions
        log['lidar_reconstruction'] = lidar_reconstructions
        return log
