import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src1/")
import torch
from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate
import fire
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
from tqdm import tqdm

def main(val_s: int=0, val_e: int=10, rollout: int=5):

    pretrained_model_name_or_path = "/data/wuzhirong/hf-models/i2vgen-xl"
    pipe = I2VGenXLPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    # pipe.to("cuda")

    print('Pipeline loaded!')

    train_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
            data_root="/data/wangxd/nuscenes/",
            height=480,
            width=720,
            max_num_frames=2,
            encode_video=None,
            encode_prompt=None,
        )
    
    # root_dir = "/data/wangxd/IJCAI25/Ablation/I2VGen-XL_roll5_item5"
    root_dir = "./test1"
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
            for ridx in tqdm(range(rollout)):
                pipeline_args = {
                    "image": load_image(first_frame_path) if ridx==0 else total_frames[-1],
                    "prompt": validation_prompt,
                    "guidance_scale": int(guidance_scale),
                    "num_frames": num_frames,
                    "num_inference_steps": 25,
                }
                frames = pipe(**pipeline_args).frames[0]
                total_frames.extend(frames if ridx==(rollout-1) else frames[:-1])
            export_to_video(total_frames, os.path.join(tgt_dir, f"{key_idx:04d}.mp4"), fps=8)
            

if __name__ == "__main__":
    fire.Fire(main)