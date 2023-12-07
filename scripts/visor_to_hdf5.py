import numpy as np
import pathlib
from glob import glob
from posixpath import basename, splitext
from tqdm import tqdm
import json
import argparse
import os

from utils.pyutil import unzip
from ldm.utils.visor_utils import mask_to_img
import h5py
import re

def json_to_h5py(path,out_path):
    seg_data = json.load(open(path,'r'))
    num = len(seg_data['video_annotations'])
    anns = [json.dumps(x) for x in tqdm(seg_data['video_annotations'])]

    seg_name = re.search(r'P\d+_\d+',path)[0]
    sparse_path = [ e for e in sparse_paths if seg_name+'.json' in e][0]
    sparse_seg_data = json.load(open(sparse_path,'r'))
    sparse_map = {ann['image']['image_path'][:-4]: ann for ann in sparse_seg_data['video_annotations']}

    gt_anns = []
    has_hands = []
    frame_nums = []
    for x in tqdm(seg_data['video_annotations']):
        ip = x['image']['image_path'][:-4]
        fn = int(ip[-10:])
        frame_nums.append(fn)
        if ip in sparse_map:
            gt_anns.append(1)
            ann = sparse_map[ip]
            input_resolution = (1920,1080)
        else:
            gt_anns.append(0)
            ann = x
            input_resolution = (854,480)
        object_keys = {'left hand': 1, 'right hand': 2}
        hand_mask = mask_to_img(ann['annotations'],object_keys,input_resolution=input_resolution,output_resolution=(456,256))
        has_left_hand = (hand_mask == 1).sum() > 0
        has_right_hand = (hand_mask == 2).sum() > 0
        has_hands.append((has_left_hand,has_right_hand))

    gt_anns = np.array(gt_anns)
    has_hands = np.array(has_hands)
    frame_nums = np.array(frame_nums)
    gt_inds = np.where(gt_anns)[0]
    gt_no_hands = np.zeros_like(gt_anns)
    consistent_with_gt = np.zeros_like(gt_anns)
    for s,e in zip(gt_inds,gt_inds[1:]):
        if has_hands[[s,e]].sum() == 0:
            gt_no_hands[s:e+1] = 1
        seg = has_hands[s:e+1]
        if np.all(seg == seg[0]):
            consistent_with_gt[s:e+1] = 1
    assert len(anns) == len(has_hands)
    assert len(anns) == len(gt_no_hands)
    assert len(anns) == len(consistent_with_gt)

    sparse_frame_nums = []
    sparse_anns = []
    for k,ann in sparse_map.items():
        fn = int(k[-10:])
        sparse_frame_nums.append(fn)
        sparse_anns.append(json.dumps(ann))

    fil = h5py.File(out_path, "w")
    dt = h5py.special_dtype(vlen=str)
    ann_dataset = fil.create_dataset('annotations',(num,),dtype=dt)
    for i,x in enumerate(anns):
        ann_dataset[i] = x
    fil.create_dataset('hands',data=has_hands)
    fil.create_dataset('gt_no_hands',data=gt_no_hands)
    fil.create_dataset('consistent_with_gt',data=consistent_with_gt)
    fil.create_dataset('frame_nums',data=frame_nums)
    sparse_ann_dataset = fil.create_dataset('sparse_annotations',(len(sparse_anns),),dtype=dt)
    for i,x in enumerate(sparse_anns):
        sparse_ann_dataset[i] = x
    fil.create_dataset('saprse_frame_nums',data=sparse_frame_nums)
    fil.close()

import hashlib
def determ_hash(x,salt):
    hex_val = hashlib.sha224((json.dumps(x)+salt).encode()).hexdigest()
    val = int(hex_val,16)
    return val
def for_this_thread(ids,num_threads,thread_num,salt=""):
    return (determ_hash(ids,salt) % num_threads) == thread_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('epic_location')
    parser.add_argument('visor_location')
    parser.add_argument('output_location')
    parser.add_argument('--num-threads',default=1,type=int)
    parser.add_argument('--thread-num',default=0,type=int)
    parser.add_argument('--salt',default="")
    args = parser.parse_args()

    EPIC_ROOT = args.epic_location
    paths = glob(f'{args.visor_location}/Interpolations-DenseAnnotations/*/*.json',recursive=True)
    sparse_paths = glob(f'{args.visor_location}/GroundTruth-SparseAnnotations/annotations/**/*.json',recursive=True)
    out_root = args.output_location
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)
    # filter for multi thread version
    print('None hash:', determ_hash(None,args.salt))
    print('all: ',len(paths))
    paths = [x for x in paths if for_this_thread(x,args.num_threads,args.thread_num,args.salt)]
    print('filtered: ',len(paths))
    for path in tqdm(paths):
        print(path)
        out_path = os.path.join(out_root,splitext(basename(path))[0] + ".hdf5")
        if os.path.exists(out_path):
            print(f'skipping {out_path}')
        else:
            json_to_h5py(path,out_path)

