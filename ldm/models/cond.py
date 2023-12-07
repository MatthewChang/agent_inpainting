from einops.einops import rearrange
from einops import repeat
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils.pyutil import to_channel_first, to_channel_last

class MultiImage(nn.Module):
    def __init__(self,first_stage_model=None):
        super(MultiImage, self).__init__()
        self.first_stage_model = first_stage_model
    def forward(self,x):
        with torch.no_grad():
            B = x.shape[0]
            # flatten into big batch
            xb = rearrange(x,'b (t c) h w -> (b t) c h w',c=3)
            encoded = self.first_stage_model.encode(xb)
            return rearrange(encoded,'(b t) c h w -> b (t c) h w',b=B)


class Inpainting(nn.Module):
    def __init__(self,first_stage_model=None,multi_img = False):
        super(Inpainting, self).__init__()
        self.first_stage_model = first_stage_model
        self.multi_img = multi_img
    def forward(self,x):
        with torch.no_grad():
            B = x['hand_images'].shape[0]
            # x['mask'] is batch x time x height x width, [-1,-1]
            mask = x['mask'][:,-1:]
            if self.multi_img:
                # hand images should be: b t h w c, [-1,1], shoud already have the masked values as -1
                xb = rearrange(x['hand_images'],'b t h w c -> (b t) c h w')
                encoded = self.first_stage_model.encode(xb)
                encoded = rearrange(encoded,'(b t) c h w -> b t c h w',b=B)
                c1 = encoded[:,-1]
                rest = rearrange(encoded[:,:-1],'b t c h w -> b (t c) h w',b=B)
                cc = torch.nn.functional.interpolate(mask, size=c1.shape[-2:])
                # keep the same initial order by stacking as [last-frame,mask,rest]
                ret = torch.cat((c1, cc,rest), dim=1).float()
            else:
                # x['hand_images'] should be batch x height x width x (time channels)
                masked_image = x['hand_images'][...,-3:]
                c1 = self.first_stage_model.encode(to_channel_first(masked_image))
                cc = torch.nn.functional.interpolate(mask, size=c1.shape[-2:])
                ret = torch.cat((c1, cc), dim=1).float()
            return ret 

class Inpainting3d(nn.Module):
    def __init__(self,first_stage_model=None,multi_img = False):
        super(Inpainting3d, self).__init__()
        self.first_stage_model = first_stage_model
        self.multi_img = multi_img
    def forward(self,x):
        with torch.no_grad():
            B = x['hand_images'].shape[0]
            # return torch.zeros((B,13,64,64),device=x['hand_images'].device)
            # x['mask'] is batch x time x height x width
            mask = x['mask']
            # add channel dim
            xb = rearrange(x['hand_images'],'b t h w c -> (b t) c h w')
            encoded = self.first_stage_model.encode(xb)
            encoded = rearrange(encoded,'(b t) c h w -> b t c h w',b=B)
            cc = torch.nn.functional.interpolate(mask, size=encoded.shape[-2:])
            # add channel dim
            cc = rearrange(cc,'b t h w -> b t 1 h w')
            # keep the same initial order by stacking as [last-frame,mask,rest]
            ret = torch.cat((encoded, cc), dim=-3).float()
            return ret

