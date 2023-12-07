from genericpath import isfile
from posixpath import basename, splitext
from utils.pyutil import to_channel_last
from cv2 import cv2
import imageio

import numpy as np
import os
from tqdm import tqdm
import torch
import h5py

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from ldm.detectron import predict_batch

from detectron2.data.datasets.pascal_voc import register_pascal_voc

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('ego_root')
parser.add_argument('out_dir')
parser.add_argument('--ids',nargs='+')
args = parser.parse_args()

_datasets_root = "datasets"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')

ego_root = args.ego_root
write_dir = args.out_dir
from glob import glob
if args.ids is not None and len(args.ids) > 0:
    vids = [f'{ego_root}/{e}.mp4' for e in args.ids]
else:
    vids = sorted(glob(f"{ego_root}/*.mp4",recursive=True))
print(f'running on {len(vids)} ', vids)

# load cfg and model
cfg = get_cfg()
cfg.merge_from_file("100doh/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
cfg.MODEL.WEIGHTS = '100doh/models/model_0529999.pth' # add model weight here
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model
import detectron2
import torchvision.transforms as trans
import torch.nn.functional as F
import torch.utils.data
def ResizeShortestEdge(size):
    return lambda x: F.interpolate(x.unsqueeze(0),scale_factor=size/min(x.shape[-2:]))[0]


import torch
class VideoReader(torch.utils.data.IterableDataset):
     def __init__(self, vid,frame_skip=1):
        super(VideoReader).__init__()
        self.reader = imageio.get_reader(vid)
        self.frame_skip = frame_skip 
     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None or worker_info.num_workers == 1:  # single-process data loading, return the full iterator
            def gen():
                for i,frame in enumerate(self.reader):
                    if i % frame_skip == 0:
                        yield np.array(frame)
            # return iter(gen())
            return gen()
         else:  # in a worker process
             print(worker_info)
             raise Exception(f'only works with one worker')


transform = trans.Compose((trans.ToTensor(),ResizeShortestEdge(256),trans.CenterCrop(256)))
predictor = DefaultPredictor(cfg)

batch_size = 20
frame_skip = 10
padding=3
for vid in tqdm(vids):
    vid_id = splitext(basename(vid))[0]
    out_path = os.path.join(write_dir,f'{vid_id}.hdf5')
    if isfile(out_path):
        continue
    loader = torch.utils.data.DataLoader(VideoReader(vid,frame_skip=frame_skip),num_workers=1,batch_size=batch_size,collate_fn=lambda x: x)
    batch_loader = iter(loader)
    all_frames = []
    labels = []
    num_batches = 0
    for batch in tqdm(batch_loader):
        num_batches += 1
        # need to switch to numpy H W C
        outputs = predict_batch(predictor,batch)
        for out,img in zip(outputs,batch):
            if len(out['instances']) > 0:
                labels.append(1)
            else:
                labels.append(0)
            all_frames.append(transform(img))

    all_frames = torch.stack(all_frames)
    mean_pixel = (all_frames.mean((2,3))*255).numpy().astype(int)
    numpy_frames = (to_channel_last(all_frames)*255).numpy().astype(np.uint8)
    encoded = [cv2.imencode('.jpg',frame)[1] for frame in tqdm(numpy_frames)]
    fil = h5py.File(out_path, "w")
    fil.create_dataset('labels',data=labels)
    fil.create_dataset('mean_pixel',data=mean_pixel)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    fd = fil.create_dataset('frames',(len(encoded),),dtype=dt)
    for i,f in enumerate(encoded):
        fd[i] = np.frombuffer(f,dtype=np.uint8)
    fil.close()

