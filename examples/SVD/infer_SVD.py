import sys
sys.path.append("/workspace/wxd/diffusers/src")

import torch
import os
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

from tqdm import tqdm
import fire

import uuid
import random


from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate


def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image


def main(val_s: int=0, val_e: int=10, rollout: int=5):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "/data2/wangxd/models/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    print('Pipeline loaded!')
    train_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
            data_root="/data/wangxd/nuscenes",
            height=480,
            width=720,
            max_num_frames=2,
            encode_video=None,
            encode_prompt=None,
        )
    
    root_dir = "/data/wangxd/IJCAI25/Ablation/SVD"
    # root_dir = "./test1"
    num_frames = 25
    os.makedirs(root_dir, exist_ok=True)
    for i in tqdm(range(val_s, val_e)): # each scene
        item = train_dataset[i] # total samples in a scene
        # if have 5 samples, cur_item_nums = 5
        key_indexs = range(5)
        for key_idx in tqdm(key_indexs):
            key_frame = item[key_idx]
            pil_videos = key_frame["instance_video"]
            validation_prompt = key_frame["instance_prompt"]

            validation_prompt = validation_prompt.split(".")[0] # prompt to long 
            
            first_frame_path = pil_videos[0]
            
            tgt_dir = os.path.join(root_dir, f"{i}")
            
            os.makedirs(tgt_dir, exist_ok=True)
            
            guidance_scale = 9
            
            total_frames = []
            motion_bucket_id = 127
            fps_id = 8
            version = "svd_xt"
            cond_aug = 0.02
            decoding_t = 3
            generator = torch.manual_seed(42)
            
            for ridx in tqdm(range(rollout)):
                image = load_image(first_frame_path) if ridx==0 else total_frames[-1]
                image = resize_image(image)
                frames = pipe(image, decode_chunk_size=decoding_t, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.1, num_frames=25).frames[0]

                total_frames.extend(frames if ridx==(rollout-1) else frames[:-1])

            export_to_video(total_frames, os.path.join(tgt_dir, f"{key_idx:04d}.mp4"), fps=8)

if __name__ == "__main__":
    fire.Fire(main)
