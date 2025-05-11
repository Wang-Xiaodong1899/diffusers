cd image_encoder
aria2c -x 16 https://hf-mirror.com/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/image_encoder/model.fp16.safetensors -o model.fp16.safetensors
cd ..
cd unet
aria2c -x 16 https://hf-mirror.com/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -o diffusion_pytorch_model.fp16.safetensors
cd ..
cd vae
aria2c -x 16 https://hf-mirror.com/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -o diffusion_pytorch_model.fp16.safetensors