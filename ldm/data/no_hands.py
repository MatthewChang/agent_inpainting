from math import e
import yaml
from glob import glob
import os
from funcy import walk,lmap
from omegaconf.omegaconf import OmegaConf
import torch.nn.functional as F
import torch
import json
from pathlib import Path
from posixpath import basename, splitext
from cv2 import cv2
from einops.einops import rearrange, repeat
import numpy as np
import PIL
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from ldm.data.masks import MixedMaskGenerator
from ldm.utils.visor_utils import mask_to_img
from ldm.util import instantiate_from_config
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import re
import h5py
from glob import glob
from os import write
from posixpath import basename
from glob import glob
from pathlib import Path


mask_gen_kwargs = {
    "irregular_proba": 1,
    "irregular_kwargs": {
        "min_times": 4,
        "max_times": 5,
        "max_width": 50,
        "max_angle": 4,
        "max_len": 100,
    },
    "box_proba": 0.3,
    "box_kwargs": {
        "margin": 0,
        "bbox_min_size": 10,
        "bbox_max_size": 50,
        "max_times": 5,
        "min_times": 1,
    },
    "segm_proba": 0,
    "squares_proba": 0,
}

class NoHands(Dataset):
    def __init__(self,
                 visor_files,
                 epic_hdf5_root,
                 data_root=None,
                 chunk_size=0,
                 size=None,
                 aux_size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 frame_skip=1,
                 num_frames=1,
                 predict=False,
                 with_hand_bg = False,
                 loss_mask_hand_bg = True,
                 mask_size = None,
                 black_hand = False,
                 background_config = None,
                 hand_mask_rate = 0,
                 random_mask_rate = 0,
                 mask_dilation=7,
                 same_random_mask_rate = 0.5,
                 stack_frames=True,
                 no_hand_mult=1,
                 diagnostic_mask=False,
                 diagnostic_diff_frame=False,
                 add_hands=True,
                 ):
        self.aux_size=aux_size
        self.mask_dilation = mask_dilation
        if self.mask_dilation%2 == 0:
            raise Exception(f'Dilation must be odd')
        self.size = size
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.black_hand = black_hand
        # min_size = 4
        # min_size = frame_skip*num_frames - (frame_skip-1)
        self.stack_frames = stack_frames
        if self.frame_skip == 'exponential':
            self.offsets = np.flip(-exponential_diffs)
            assert self.num_frames == 4
        else:
            self.offsets = np.arange(num_frames)*frame_skip
        min_size = self.offsets[-1]+1
        self.with_hand_bg = with_hand_bg
        self.mask_size = mask_size
        self.hand_mask_rate = hand_mask_rate
        self.random_mask_rate = random_mask_rate
        self.same_random_mask_rate = same_random_mask_rate
        self.diagnostic_mask = diagnostic_mask
        self.add_hands = add_hands
        self.diagnostic_diff_frame = diagnostic_diff_frame
        self.loss_mask_hand_bg = loss_mask_hand_bg
        if background_config is not None:
            self.background_dataset = instantiate_from_config(background_config)
        print(min_size, self.offsets)
        # treat as a folder
        if isinstance(visor_files,str):
            # if splitext(visor_files)
            if splitext(visor_files)[1] == '.txt':
                basenames = open(visor_files,'r').read().split('\n')
                visor_files = sorted([os.path.join(data_root,bn) for bn in basenames if len(bn) > 0])
            else:
                visor_files = sorted(glob(f'{visor_files}/*.hdf5'))
        self.segmentation_files = []
        for x in visor_files:
            try:
                self.segmentation_files.append(h5py.File(x,'r'))
            except Exception as e:
                print(f'error reading {x}')
                raise e
        epic_data_files = glob(f'{epic_hdf5_root}/*.hdf5')
        self.epic_data = {}
        for path in epic_data_files:
            name = basename(splitext(path)[0])
            self.epic_data[name] = h5py.File(path,'r')
        self.no_hands_samples = []
        self.has_hands_samples = []
        for i,fil in enumerate(self.segmentation_files):
            hands = fil['hands'][:].max(-1)
            gt_no_hands = fil['gt_no_hands'][:].max(-1)
            consistent_with_gt = fil['consistent_with_gt'][:].max(-1)
            # only consider samples consistenet with GT i.e. the presence
            # of masks matches the nearest ground truth annotation before and after
            hands = hands & consistent_with_gt
            gt_no_hands = gt_no_hands & consistent_with_gt
            kernel = np.ones((min_size,))/min_size
            # get start points of sequences with no hands
            has_no_hands_min = np.convolve(~gt_no_hands,kernel,'valid') == 0
            no_hands_start_inds = np.where(has_no_hands_min)[0]
            no_hand_samples = np.stack((np.ones_like(no_hands_start_inds)*i,no_hands_start_inds)).T
            self.no_hands_samples.append(no_hand_samples)
            # get start points of sequences with do have hands
            has_hands_min = np.convolve(~hands,kernel,'valid') == 0
            hands_start_inds = np.where(has_hands_min)[0]
            hand_samples = np.stack((np.ones_like(hands_start_inds)*i,hands_start_inds)).T
            self.has_hands_samples.append(hand_samples)
        self.no_hands_samples = np.concatenate(self.no_hands_samples,0)
        self.has_hands_samples = np.concatenate(self.has_hands_samples,0)
        self.predict = predict
        self.mask_generator = MixedMaskGenerator(**mask_gen_kwargs)
        if background_config is not None:
            inds = np.arange(len(self.background_dataset))
            bg_samples = np.stack((np.ones_like(inds)*-1,inds),1)
            self.no_hands_samples = np.concatenate((self.no_hands_samples,bg_samples))
            print("added bgs:", len(bg_samples))
        print(len(self.no_hands_samples),len(self.has_hands_samples))
        self.no_hand_mult = no_hand_mult
        if self.with_hand_bg:
            self.no_hands_samples = np.tile(self.no_hands_samples,(self.no_hand_mult,1))
            print('mult no hand: ',len(self.no_hands_samples))
            self.bg_samples = np.concatenate((self.no_hands_samples,self.has_hands_samples),0)
        else:
            self.bg_samples = self.no_hands_samples
        
    def __len__(self):
        if self.predict:
            return len(self.has_hands_samples)
        else:
            return len(self.bg_samples)

    def process_image(self,img,size=None,no_scale=False):
        if not size:
            size = self.size
        assert size is not None

        # cv2 decoes as bgr, so flip channels
        img = img[:,:,::-1]

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((size, size), resample=self.interpolation)
        if no_scale:
            return np.array(image)
        else:
            return (np.array(image) / 127.5 - 1.0).astype(np.float32)

    # lookup and return image data from annotation json
    def get_img(self,ann_dat):
        video = ann_dat['image']['video']
        frame_ind = int(re.match('.*frame_(\d+).png',ann_dat['image']['name'])[1])-1
        img_dat = self.epic_data[video]['frames'][frame_ind]
        img = cv2.imdecode(img_dat,cv2.IMREAD_COLOR)
        return img

    # get mask and images from annotation data
    def get_hands(self,ann_dat):
        img = self.get_img(ann_dat)
        object_keys = {'left hand': 1, 'right hand': 1}
        # input masks are generated in 480 x 854
        # shapes are width,height
        input_resolution = (854,480)
        hand_mask = mask_to_img(ann_dat['annotations'],object_keys,input_resolution=input_resolution,output_resolution=(img.shape[1],img.shape[0]))
        return img,hand_mask

    def get_background(self,seq,num):
        if seq == -1:
            imgs = self.background_dataset[num]
            # flip channels because processing expects bgr
            imgs = imgs[...,::-1].copy()
            bg_hand_mask = np.zeros_like(imgs)
        else:
            nums = num+self.offsets
            ann_dats = [json.loads(self.segmentation_files[seq]['annotations'][num]) for num in nums]
            imgs,bg_hand_mask = zip(*[self.get_hands(x) for x in ann_dats])
        return imgs,bg_hand_mask

    def __getitem__(self, i):
        # source for hands
        hand_source = self.has_hands_samples[random.randint(0,len(self.has_hands_samples)-1)]
        seq,start = hand_source
        hand_nums = start+self.offsets
        ann_hand_dats = [json.loads(self.segmentation_files[seq]['annotations'][num]) for num in hand_nums]
        # ann_dat_hand = json.loads(self.segmentation_files[seq]['annotations'][start])
        hand_img,hand_mask = zip(*[self.get_hands(x) for x in ann_hand_dats])
        hand_img = np.stack(hand_img)
        hand_mask = np.stack(hand_mask)
        hand_mask_proc = np.array([self.process_image(e,no_scale=True) for e in hand_mask])

        mask_hand = random.random() < self.hand_mask_rate
        use_random_mask = random.random() < self.random_mask_rate
        use_same_random_mask = random.random() < self.same_random_mask_rate

        if mask_hand:
            mask = hand_mask_proc[...,0] > 0
        else:
            mask = np.zeros_like(hand_mask_proc[...,0]).astype(bool)
        # dilate mask to avoid feather of hand pixels
        dilation_kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
        dilate = lambda x: cv2.dilate(x.astype(np.uint8),dilation_kernel,iterations=1)
        mask = np.stack([dilate(x).astype(bool) for x in mask])

        if self.predict:
            ret = {}
            # t h w c
            hand_images = np.stack(list(map(self.process_image,hand_img)))
            if mask_hand:
                if self.diagnostic_mask:
                    assert self.num_frames == 2
                    mask = mask[-1]
                    mask = np.stack((1-mask,mask.astype(float)))
                    if not self.diagnostic_diff_frame:
                    # make it the same iamge
                        hand_images[0] = hand_images[1]
                    hand_images[mask > 0.5] = [-1,-1,-1]
                else:
                    hand_images[mask] = [-1,-1,-1]
            ret['image'] = np.zeros(hand_images.shape[1:])
            ret['mask'] = mask.astype(float) * 2 - 1
            ret['loss_mask'] = np.zeros(hand_images.shape[1:-1]).astype(bool)
            if self.stack_frames:
                ret['hand_images'] = rearrange(hand_images,'t h w c -> h w (t c)')
            else:
                ret['hand_images'] = hand_images
            return ret
        else:
            # background source
            seq,num = self.bg_samples[i]
            imgs,bg_hand_mask = self.get_background(seq,num)
            imgs = np.stack(imgs)
            bg_hand_mask = np.stack(bg_hand_mask)
            bg_hand_mask = np.stack(lmap(dilate,bg_hand_mask))
            # bg_hand_mask = cv2.dilate(bg_hand_mask.astype(np.uint8),dilation_kernel,iterations=2)

            if mask_hand:
                imgs[(bg_hand_mask[:,:,:,0] > 0)] = [0,0,0] 
            ret = {}
            ret['image'] = self.process_image(imgs[-1].copy())
            # crop center
            if self.loss_mask_hand_bg:
                loss_mask = self.process_image(bg_hand_mask[-1].copy()*255) > 0
                (self.process_image(bg_hand_mask[-1].copy()*255) > 0).shape
            else: 
                loss_mask = np.zeros_like(ret['image']).astype(bool)
            if mask_hand:
                # now loss mask is h w c
                mask = mask | loss_mask[:,:,0]
            # get right shape for interpolate (b c w h)
            loss_mask = torch.tensor(loss_mask[None,None,:,:,0])
            # resize
            if self.mask_size is not None:
                loss_mask_scaled = F.interpolate(loss_mask.float(),tuple(self.mask_size),mode='area')
            else:
                loss_mask_scaled = loss_mask

            hand_img_proc = np.array(lmap(self.process_image,hand_img))
            imgs = np.array(lmap(self.process_image,imgs))
            # get 1 only for super pixels which has no overlap with gt mask
            loss_mask_scaled = ~(loss_mask_scaled > 0)
            ret['loss_mask'] = loss_mask_scaled[0,0].numpy()
            # paste hand over original img
            if self.diagnostic_mask:
                assert self.num_frames == 2
                mask_shape = imgs.shape[1:3]
                m1 = self.mask_generator(mask_shape).astype(bool) | loss_mask[0,0].numpy()
                mask = np.concatenate((1-m1,m1))
                if not self.diagnostic_diff_frame:
                    # make it the same iamge
                    imgs[0] = imgs[1]
                imgs[mask > 0.5] = [-1,-1,-1]
            else:
                if self.add_hands:
                    if mask_hand:
                        # black in the [-1,1] color space
                        imgs[mask] = [-1,-1,-1]
                    else:
                        imgs[hand_mask_proc == 1] = hand_img_proc[hand_mask_proc==1]
                if use_random_mask:
                    mask_shape = imgs.shape[1:3]
                    masks = [self.mask_generator(mask_shape) for _ in range(len(imgs))]
                    if use_same_random_mask:
                        masks = repeat(masks[0],'1 h w -> n h w',n = len(imgs))
                    else:
                        masks = np.concatenate(masks,0)
                    imgs[masks > 0] = [-1,-1,-1]
                    mask = mask | (masks > 0)
            ret['mask'] = mask.astype(float) * 2 - 1
            if self.stack_frames:
                ret['hand_images'] = rearrange(imgs,'b h w c -> h w (b c)')
            else:
                ret['hand_images'] = imgs
            if random.random() < self.flip_p:
                ret['image'] = np.flip(ret['image'],-2).copy()
                ret['hand_images'] = np.flip(ret['hand_images'],-2).copy()
                ret['loss_mask'] = np.flip(ret['loss_mask'],-1).copy()
                ret['mask'] = np.flip(ret['mask'],-1).copy()
            return ret
