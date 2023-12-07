import pathlib
import re
from posixpath import basename
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('epic_root')
parser.add_argument('out_dir')
args = parser.parse_args()

root = args.epic_root
out = args.out_dir
pathlib.Path(out).mkdir(parents=True, exist_ok=True)
scenes = sorted(glob(f"{root}/*",recursive=True))
print(scenes)
all_sub_scenes = []
for sc in scenes:
    sub_scenes = glob(f'{sc}/rgb_frames/*')
    all_sub_scenes += sub_scenes
print(all_sub_scenes)
print(len(all_sub_scenes))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst, last chunk may be smaller"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

import h5py
all_sub_scenes = sorted(all_sub_scenes)
for ss_path in tqdm(all_sub_scenes):
    ss = basename(ss_path)
    frame_numbers = []
    frames = []
    samples = sorted(glob(f'{ss_path}/*.jpg'))
    for sam in tqdm(samples):
        frame_numbers.append(int(re.match(r'frame_(\d+).jpg',basename(sam))[1]))
        frames.append(open(sam, 'rb').read())
    fp = f'{out}/{ss}.hdf5'
    fil = h5py.File(fp, "w")
    fil.create_dataset('frame_numbers',data=frame_numbers)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    fd = fil.create_dataset('frames',(len(frames),),dtype=dt)
    for i,f in enumerate(frames):
        fd[i] = np.frombuffer(f,dtype=np.uint8)
    fil.close()

