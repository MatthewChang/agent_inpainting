import os
import bisect
from cv2 import cv2
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import h5py
from glob import glob
from posixpath import splitext
import numpy as np

exponential_diffs = np.array([  0,   1,   2,   3,   4,   5,   7,  12,  20,  34,  59, 100, 171])

class Backgrounds(Dataset):
    def __init__(self,
                 files,
                 data_root='./',
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 frame_skip=1,
                 padding=1,
                 num_frames=1):
        self.size = size
        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        if frame_skip == 'exponential':
            self.offsets = exponential_diffs
        else:
            self.offsets = np.arange(num_frames)*frame_skip
        min_size = self.offsets[-1]+1
        # treat as a folder
        if isinstance(files,str):
            # if splitext(files)
            if splitext(files)[1] == '.txt':
                basenames = open(files,'r').read().split('\n')
                files = sorted([os.path.join(data_root,bn) for bn in basenames if len(bn) > 0])
            else:
                files = sorted(glob(f'{files}/*.hdf5'))
        self.sources = []
        for x in files:
            try:
                self.sources.append(h5py.File(x,'r'))
            except Exception as e:
                print(f'error reading {x}')
                raise e
        self.no_hands_samples = []
        self.sorted_no_hands = {}
        for i,fil in enumerate(self.sources):
            black_img = fil['mean_pixel'][:].mean(-1) <= 10
            # if(black_img.sum()/len(black_img) > 0.3):
                # raise Exception(f'More than 0.3 black {files[i]}')
            labels = fil['labels'][:]
            with_hand_padded = np.convolve(np.array(labels), np.ones((padding*2+1,)),mode='same').astype(int) > 0
            # np.where(~with_hand_padded)
            # consider black images as having hand (i.e. don't include)
            has_hand = with_hand_padded | black_img
            # get start points of sequences with no hands
            kernel = np.ones((min_size,))/min_size
            has_no_hands_min = np.convolve(has_hand,kernel,'valid') == 0
            no_hands_start_inds = np.where(has_no_hands_min)[0]
            no_hand_samples = np.stack((np.ones_like(no_hands_start_inds)*i,no_hands_start_inds)).T
            self.no_hands_samples.append(no_hand_samples)
        self.no_hands_samples = np.concatenate(self.no_hands_samples,0)


    def nearest_no_hands(self,vid,fn):
        fns = self.sorted_no_hands[vid]
        ind = bisect.bisect_left(fns,fn)
        v1 = fns[min(ind,len(fns)-1)]
        v2 = fns[min(ind+1,len(fns)-1)]
        if abs(fn-v1) < abs(fn-v2):
            return v1
        else:
            return v2

        
    def __len__(self):
        return len(self.no_hands_samples)

    # def process_image(self,img,size=None):
        # if not size:
            # size = self.size
        # assert size is not None

        # # flip channels because the no_hands class expects bgr
        # img = img[:,:,::-1]

        # crop = min(img.shape[0], img.shape[1])
        # h, w, = img.shape[0], img.shape[1]
        # img = img[(h - crop) // 2:(h + crop) // 2,
              # (w - crop) // 2:(w + crop) // 2]

        # image = Image.fromarray(img)
        # image = image.resize((size, size), resample=self.interpolation)
        # return (np.array(image) / 127.5 - 1.0).astype(np.float32)

    # RGB frames
    def __getitem__(self, i):
        pid,start = self.no_hands_samples[i]
        hand_nums = start+self.offsets
        frames = [cv2.imdecode(self.sources[pid]['frames'][x],cv2.IMREAD_COLOR) for x in hand_nums]
        if random.random() < self.flip_p:
            frames = [np.flip(x,1).copy() for x in frames]
        frames = np.stack(frames)
        return frames
        # return frames
