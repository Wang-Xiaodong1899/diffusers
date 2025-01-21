import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
sys.path.append("/home/user/wangxd/diffusers/src/")

import torch
from diffusers import AutoencoderKLCogVideoX

from diffusers.utils import load_image, export_to_video
from tqdm import tqdm

from diffusers.video_processor import VideoProcessor
video_processor = VideoProcessor(vae_scale_factor=8)

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

# device = "cpu"
device = "cuda"

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )
vae.to(device)
# vae.enable_slicing()
# vae.enable_tiling()
print('Pipeline loaded!')

import torch
import torch.fft as fft

def fourier_filter(x, scale=0., d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, F, H, W = x_freq.shape
    mask = torch.ones((B, C, F, H, W),device=x_freq.device)

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
            if d_square < 2 * d_s:
                mask[..., h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered

def fourier_filter_3d(x, scale=0., d_s=0.25, d_t=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

    B, C, T, H, W = x_freq.shape
    mask = torch.ones((B, C, T, H, W),device=x_freq.device)

    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2 + d_s / d_t *(2 * t / T - 1) ** 2
                if d_square < 3*d_s:
                    mask[..., t, h, w] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-3, -2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered

# img_path = "/home/user/wuzhirong/Projects/diffusers/wzr_fft/rollout0/0001.png"

# img_root = "./rollout0"
# img_names = sorted(os.listdir(img_root))

img_root = "/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/"
img_names = sorted(os.listdir(img_root))[2500:2500+13*2:2]
# img_names = [
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281638112460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281638612460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281639162460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281639662460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281640162460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281640662460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281641162460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281641662460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281642162460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281642662460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643162629.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281643662460.jpg",
#     "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281644162460.jpg",
# ]
os.makedirs("origin_imgs", exist_ok=True)
# for im in img_names:
#     os.system(f"cp {os.path.join(img_root, im)} origin_imgs/")

imgs = [load_image(os.path.join(img_root, img_names[i])) for i in range(len(img_names))]
imgs = video_processor.preprocess(imgs, height=480, width=720).to(device=device,dtype=torch.float16) # [F,C,H,W]
imgs = imgs.unsqueeze(0).permute(0, 2, 1 ,3,4) # [B, C, F, H, W]

d_s = 0.1
d_t = 0.1

with torch.no_grad():
    latents = []
    for i in tqdm(range(len(img_names))):
        latent = vae.encode(imgs[:,:,i:i+1]).latent_dist.sample()
        latents.append(latent)
    latents = torch.cat(latents, dim=2)
    
    print(latents.shape)

    # latents = fourier_filter(latents, d_s=d_s)
    latents = fourier_filter_3d(latents, d_s=d_s, d_t=d_t)

    rec_imgs = []
    for i in tqdm(range(len(img_names))):
        rec_img = vae.decode(latents[:,:,i:i+1]).sample
        rec_imgs.append(rec_img)
    rec_imgs = torch.cat(rec_imgs,dim=2)
    print(rec_imgs.shape)

video = video_processor.postprocess_video(video=rec_imgs, output_type="pil")[0]
for i, img in enumerate(video):
    img.save(os.path.join("./imgs", img_names[i].split('.')[0]+f"_fft_{d_s}_{d_t}.jpg"))
export_to_video(video, "test0.mp4", fps=1)
