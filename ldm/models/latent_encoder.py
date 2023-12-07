import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ResnetFeatures(nn.Module):
    def __init__(self,normalize=True):
        super(ResnetFeatures, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(resnet18.children())[:-1])
        self.normalize = normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_trans = transforms.Normalize(mean=mean, std=std)
        # self.head = nn.Linear(512,64)
    def forward(self,x):
        if self.normalize:
            # check if 0-255 or 0-1, maybe do it by checking types
            # import pdb; pdb.set_trace()
            assert x.min() >= 0 and x.max() <= 1
            x = self.norm_trans(x)
        orig_shape = x.shape
        x = x.view(-1,*orig_shape[-3:])
        x = self.model(x)
        # reshape initial dims
        x = x.view(*orig_shape[:-3],-1)
        # x = self.head(x)
        return x

class ResnetLatnetEncoder(nn.Module):
    def __init__(self,latent_size=512,first_stage_model=None):
        super(ResnetLatnetEncoder, self).__init__()
        if latent_size == 512:
            self.model = ResnetFeatures(normalize=True)
        else:
            self.model = nn.Sequential(ResnetFeatures(normalize=True),nn.Linear(512,latent_size))
        self.first_stage_model = first_stage_model
    def forward(self,x):
        img = x[:,:3]
        next_img = x[:,3:]
        with torch.no_grad():
            img_small = F.interpolate(img,(128,128),mode='bilinear')
            encoded = self.first_stage_model.encode(img_small)
        img_resnet,next_img_resnet = [(x+1)/2 for x in (img,next_img)]
        diffs = self.model(next_img_resnet)- self.model(img_resnet)
        # one latent_size-d token
        diffs = diffs.unsqueeze(1)
        return {'c_concat': [encoded],'c_crossattn': [diffs]}

class LatnetEncoder(nn.Module):
    def __init__(self,image_size=16,in_channels=8):
        super(LatnetEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
    def forward(self,x):
        return None

if __name__ == '__main__':
    model = ResnetFeatures()
    model.cuda()
    inp = torch.zeros(5,3,224,224)
    model(inp.cuda()).shape
