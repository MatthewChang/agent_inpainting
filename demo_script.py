from functools import partial
import numpy as np
from funcy import lmap
import os
from cv2 import cv2
from einops.einops import rearrange
from imageio.core.functions import imwrite, imread
from omegaconf import OmegaConf
from main import instantiate_from_config
import torch

from utils.cv2 import maximal_crop_to_shape
from utils.pyutil import to_channel_last

import torch.utils.data

import argparse

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="")
parser.add_argument("target_folder")
parser.add_argument("--dilation", default=7,type=int)
args = parser.parse_args()

args.model = 'inpaint_4f_best'
is_factored = args.model in ['inpaint_4f_factored']
with_extra_mask = False
process_type = 'unset'
img_size = (256,256)
checkpoint = ""
config_source = ""

process_type = 'inpaint_multi'
checkpoint = 'models/nohands_weights.ckpt'
config_source = "configs/inpaint_4frame.yaml"

frame_packing = False
# config = OmegaConf.load("configs/latent-diffusion/inpaint_finetune3d_4f_fixed.yaml")
config = OmegaConf.load(config_source)
model = instantiate_from_config(config.model)
model.load_state_dict(
    torch.load(
        checkpoint,
        map_location="cuda:0",
    )["state_dict"]
)
print("loading: ",checkpoint)
model.cuda()

# Switch to use EMA weights
model.model_ema.store(model.model.parameters())
model.model_ema.copy_to(model.model)
print("USING EMA")

from glob import glob
from imageio import imread
rgb = sorted(glob(f"{args.target_folder}/raw/*.jpg",recursive=True))
masks = sorted(glob(f"{args.target_folder}/masks/*.png",recursive=True))
rgb = np.array(list(map(imread,rgb)))
masks = np.array(list(map(imread,masks)))
assert masks.shape[-1] == 3

# crop and dilate
from functools import partial
crop = lambda x: maximal_crop_to_shape(x,(256,256))
crop_ims = partial(lmap,crop)
im = crop_ims(rgb)
mask = crop_ims(masks)
dilation_kernel = np.ones((args.dilation, args.dilation), np.uint8)
mask = [cv2.dilate(ma,dilation_kernel) for ma in mask]
im,mask = lmap(np.stack,(im,mask))
# binarize mask
mask = mask > 10

# select slices with each of the 4 images as the inpainting target
indices = [[1,2,3,0],[0,2,3,1],[0,1,3,2],[0,1,2,3]]
batched_im = im[indices]
batched_mask = mask[indices]

data = {'images':torch.tensor(batched_im), 'masks':torch.tensor(batched_mask)}
# add batch dimension
# data = {k: v.unsqueeze(0) for k,v in data.items()}

# convert to cuda, tailor based on your hardware
if isinstance(data,dict): data = {k: v.cuda().float() for k, v in data.items()}
# formatting for non-multi img conditioning
masked = data['images']*(1-data['masks'])
masked_normalized = (masked/127.5-1)
norm_mask = data['masks'][...,0]*2-1
cond_in = {'hand_images':masked_normalized,'mask':norm_mask}
# embed to  latent space
cond = model.cond_stage_model(cond_in)
# denoise predictions
ddim_samples, z_denoise_row = model.sample_log( cond=cond, batch_size=cond.shape[0], ddim=True, ddim_steps=200, eta=1)
# lift from latent to pixel space
decoded = model.first_stage_model.decode(ddim_samples)
rgb = to_channel_last(((decoded + 1) * 128).cpu().numpy())
rgb = np.clip(rgb, 0, 255).astype(np.uint8)

# copy over gt from unmasked region
ma = (data['masks'][:,-1] > 0).int().cpu().numpy()
raw = data['images'][:,-1]
rgb = raw.cpu().numpy()*(1-ma) + rgb*ma
out_masks = ma*255

# rearrange and output
out_im = np.stack((raw.cpu(),out_masks,rgb))
out_im = rearrange(out_im,'k t h w c -> (t h) (k w) c')
imwrite('output.png',out_im.astype(np.uint8))
