import sys
sys.path.append("/workspace/wxd/diffusers/src")

import torch
import os
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

import uuid
import random

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


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/volsparse3/wxd/models/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

motion_bucket_id: int = 127
fps_id: int = 8
version: str = "svd_xt"
cond_aug: float = 0.02
decoding_t: int = 3
device: str = "cuda"

generator = torch.manual_seed(42)
output_folder = "./test_SVD"
os.makedirs(output_folder, exist_ok=True)
base_count = 0
video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

image = Image.open("frame_0001.png")
image = resize_image(image)

frames = pipe(image, decode_chunk_size=decoding_t, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.1, num_frames=25).frames[0]
export_to_video(frames, video_path, fps=fps_id)
