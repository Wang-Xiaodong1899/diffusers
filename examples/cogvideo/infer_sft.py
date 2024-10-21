import os
import sys
sys.path.append("/root/autodl-tmp/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

pretrained_model_name_or_path = "/root/autodl-fs/models/CogVideoX-2b"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
        torch_dtype=torch.float16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    torch_dtype=torch.float16,
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        "/root/autodl-fs/ckpt/cogvideo/cogvideox-D4-clean-image-sft/checkpoint-1000",
        subfolder="transformer",
        # "/root/autodl-fs/CogVidx-2b-I2V-base-transfomer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')

while True:
    image_path = input("image_path: ")
    validation_prompt = input("prompt: ")
    guidance_scale = input("cfg: ") # 6
    pipeline_args = {
        "image": load_image(image_path),
        "prompt": validation_prompt,
        "guidance_scale": int(guidance_scale),
        "use_dynamic_cfg": True,
        "height": 480,
        "width": 720,
        "num_frames": 33
    }
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    frames = pipe(**pipeline_args).frames[0]
    export_to_video(frames, f"{name_prefix}_cfg_{guidance_scale}.mp4", fps=8)