import spaces
import torch
import os
from glob import glob
from pathlib import Path
from typing import Optional

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

import uuid
import random


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/volsparse3/wxd/models/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

motion_bucket_id: int = 127
fps_id: int = 6
version: str = "svd_xt"
cond_aug: float = 0.02
decoding_t: int = 3
device: str = "cuda"

generator = torch.manual_seed(42)
output_folder = "/test_SVD"
os.makedirs(output_folder, exist_ok=True)
base_count = 0
video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

frames = pipe(image, decode_chunk_size=decoding_t, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.1, num_frames=25).frames[0]
export_to_video(frames, video_path, fps=fps_id)
