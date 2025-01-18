import sys
sys.path.append("/home/user/wangxd/diffusers/examples/cogvideo")


from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy
import os
from tqdm import tqdm
from einops import repeat
import fire
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


# validate FVD on Nuscene val 150 samples
# default 8 frames
# default image size 256x256

# NOTE:
# to ensure accurate
# we load the video following the time steps

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from common import image2arr, pil2arr, mp4toarr, image2pil, json2data


from fvd import load_fvd_model, compute_fvd, compute_fvd_1

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate

# DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'

import json


def main(
        tgt_dirs=[
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s0-e10','/data/wangxd/ckpt/cogvideox-fbf-rec/s10-e20',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s20-e30','/data/wangxd/ckpt/cogvideox-fbf-rec/s30-e40',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s40-e50','/data/wangxd/ckpt/cogvideox-fbf-rec/s50-e60',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s60-e70','/data/wangxd/ckpt/cogvideox-fbf-rec/s70-e80',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s80-e90','/data/wangxd/ckpt/cogvideox-fbf-rec/s90-e100',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s100-e110','/data/wangxd/ckpt/cogvideox-fbf-rec/s110-e120',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s120-e130','/data/wangxd/ckpt/cogvideox-fbf-rec/s130-e140',
            '/data/wangxd/ckpt/cogvideox-fbf-rec/s140-e150',
            ],
        version = None,
        split='val',
        num_frames=8,
        i3d_device='cuda:2',
        eval_frames=25,
        skip_gt = 0,
        skip_pred = 0,
        val_array = [(0,10),(10, 20), (20,30), (30, 40), (40, 50), (50,60), (60, 70),(70, 80), (80,90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)],
):
    # [40, 50], [60, 70]

    # if version is None:
    #     version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dirs}')
    # load fvd model
    i3d_model = load_fvd_model(i3d_device)
    print('loaded i3d_model')

    # read real video
    # meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    # files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'
    
    # 10 scenes * ~35 keyframes
    real_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
        data_root="/data/wangxd/nuscenes/",
        height=480,
        width=720,
        max_num_frames=25,
        encode_video=None,
        encode_prompt=None,
    )

    # read syn videos
    syn_videos = []
    for tgt_dir in tgt_dirs:
        video_files = os.listdir(tgt_dir)
        for file in tqdm(video_files):
            if file.split('.')[-1] == 'mp4':
                video_arr = mp4toarr(os.path.join(tgt_dir, file))
                # print(video_arr[:eval_frames].shape)
                # import pdb; pdb.set_trace()
                syn_videos.append(video_arr[:eval_frames]) # only load eval_frames
    
    print(f"sync video num: {len(syn_videos)}")
    syn_videos = np.array(syn_videos)
    print(f'syn shape {syn_videos.shape}')
    
    
    real_videos = []
    for (val_s, val_e) in val_array:
        for i in tqdm(range(val_s, val_e)):
            item = real_dataset[i]
            for key_idx in tqdm(range(len(item))):
                key_frame = item[key_idx]
                pil_videos = key_frame["instance_video"]
                pil_videos = [im.resize((720, 480)) for im in pil_videos]
                real_videos.append(pil_videos[:eval_frames])
    
    print(f"real video num: {len(real_videos)}")
    # real_videos = []
    # for item in tqdm(meta_data):
    #     sce = item['scene']
    #     files = os.listdir(os.path.join(gt_dir, sce))
    #     for file in files:
    #         if file.split('.')[-1] == 'mp4':
    #             video_arr = mp4toarr(os.path.join(gt_dir, sce, file))
    #             sample_frames = video_arr[::skip_gt+1]
    #             # if len(sample_frames)<eval_frames:
    #             #     sample_frames = list(sample_frames) + [sample_frames[-1]] * (eval_frames-len(sample_frames))
    #             #     print(len(sample_frames), eval_frames-len(sample_frames))
    #             real_videos.append(sample_frames[:eval_frames]) # only load eval_frames

    # N T H W C
    real_videos = np.array(real_videos)
    print(f'real shape {real_videos.shape}')

    # fvd_score = compute_fvd(real_videos, syn_videos, i3d_model, i3d_device)
    fvd_score = compute_fvd_1(real_videos, syn_videos, i3d_model, i3d_device)
    
    print('Save Done!')

    print(f'{version} AVG FVD: {fvd_score}')

if __name__ == '__main__':
    fire.Fire(main)