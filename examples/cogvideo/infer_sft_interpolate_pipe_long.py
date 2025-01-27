import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d_interpolate import CogVideoXTransformer3DModel
# from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_interpolate import CogVideoXImageToVideoPipeline

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate

from tqdm import tqdm

from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file
import fire

import cv2
import argparse

def extract_frames(mp4_path, output_dir, fps):
    """
    Extract frames from an MP4 file at a specified FPS and save them to an output directory.

    Args:
        mp4_path (str): Path to the MP4 file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int): Frames per second to extract.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(mp4_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {mp4_path}")
        return

    # Get the original video's frame rate
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save every nth frame based on the desired FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    # print(f"Extracted {saved_count} frames to {output_dir}")

def main(val_s: int=0, val_e: int=10, num_frames: int=5, start_frame: int=2, item_per_scene: int=5):

    pretrained_model_name_or_path = "/home/user/wangxd/diffusers/CogVideoX-2b"

    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
            torch_dtype=torch.float16,
        )

    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
        torch_dtype=torch.float16,
    )

    transformer = CogVideoXTransformer3DModel.from_pretrained(
            "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-fps10-f13-1202-inherit1022/checkpoint-1000",
            subfolder="transformer",
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

    pipe = CogVideoXImageToVideoPipeline(**components, inject=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print('Pipeline loaded!')

    # dirs = ["validation_792_left", "validation_795_go_straight", "validation_793_wait"]
    # prompts = ["turn left", "go straight", "wait"]

    train_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
            data_root="/data/wangxd/nuscenes/",
            height=480,
            width=720,
            max_num_frames=2,
            encode_video=None,
            encode_prompt=None,
        )
    
    if val_s == 0 and val_e == 10:
        root_dir = os.path.join(f"/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD")
    else:
        root_dir = os.path.join(f"/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s{val_s}-e{val_e}")
    
    global_key_idx = 0 # iterpolate results corresponding to generated samples
    
    mp4_files = list(os.listdir(root_dir))
    mp4_files = [a for a in mp4_files if a.endswith(".mp4")]
    mp4_files.sort()

    for i in tqdm(range(val_s, val_e)): # each scene
        item = train_dataset[i] # total samples in a scene
        cur_item_nums = 0 # NOTE if already generate x samples for a scene, add to cur_item_nums, and also add to item_per_scene
        cur_item_nums = 5 #
        # if have 5 samples, cur_item_nums = 5
        key_indexs = [5, 9, 13, 17, 21, 25, 29, min(33, len(item)-1)]
        for key_idx in tqdm(range(len(item))):
            if key_idx in key_indexs:
                pass # need generate
            else:
                global_key_idx = global_key_idx + 1
                continue
            cur_item_nums = cur_item_nums + 1 # practical inference items
            
            # if cur_item_nums > item_per_scene:
            #     global_key_idx = global_key_idx + 1
            #     continue
            
            key_frame = item[key_idx]
            # pil_videos = key_frame["instance_video"]
            validation_prompt = key_frame["instance_prompt"]
            
            gen_dir = os.path.join(root_dir, f"{global_key_idx}", "generate")
            
            end_frame = start_frame + num_frames
            # original: interpolation: 0-1, 1-2 -> 25帧
            # interpolation-f3-
            tgt_dir = os.path.join(root_dir, f"interpolation-f{start_frame}-e{end_frame}")
            
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(tgt_dir, exist_ok=True)
            
            prefix = (
                validation_prompt[:25]
                .replace(" ", "_")
                .replace(" ", "_")
                .replace("'", "_")
                .replace('"', "_")
                .replace("/", "_")
            )
            
            
            # MP42images
            # mp4_path = os.path.join(root_dir, f"validation_video_{global_key_idx}_0_{prefix}.mp4")
            # extract_frames(mp4_path, gen_dir, fps=8)
            
            image_paths = list(os.listdir(gen_dir))
            image_paths = [a for a in image_paths if a.endswith(".png") or a.endswith(".jpg")]
            image_paths.sort()
            
            image_paths = image_paths[start_frame:start_frame+num_frames]
            
            guidance_scale = 6
            
            total_frames = []
            for idx in range(len(image_paths)-1):
                image_path = image_paths[idx]
                last_image_path = image_paths[idx+1]
                pipeline_args = {
                    "image": load_image(os.path.join(gen_dir, image_path)),
                    "last_image": load_image(os.path.join(gen_dir, last_image_path)),
                    "prompt": validation_prompt,
                    "guidance_scale": int(guidance_scale),
                    "use_dynamic_cfg": True,
                    "height": 480,
                    "width": 720,
                    "num_frames": 13
                }
                frames = pipe(**pipeline_args).frames[0]
                # import pdb; pdb.set_trace()
                # print(len(frames))
                if idx == 0:
                    total_frames.extend(frames[3:-1])
                elif idx == (len(image_paths)-2):
                    total_frames.extend(frames[3:])
                else:
                    total_frames.extend(frames[4:])
            # name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
            # print(len(total_frames))
            export_to_video(total_frames, os.path.join(tgt_dir, f"{global_key_idx}_cfg_{guidance_scale}_interpolate_test_1k.mp4"), fps=8)

            global_key_idx = global_key_idx + 1
            
            # import pdb; pdb.set_trace()
            

if __name__ == "__main__":
    fire.Fire(main)