import os
import numpy as np
from PIL import Image

def add_random_noise_to_images(input_dir, output_dir, noise_level=30, seed=None):
    """
    给目录中的所有图像添加同样的随机噪声并保存。

    Args:
        input_dir (str): 输入图像目录。
        output_dir (str): 输出图像目录。
        noise_level (int): 噪声强度（值越大，噪声越明显）。
        seed (int, optional): 随机数种子，确保可复现性。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    if not image_files:
        print("No images found in the input directory.")
        return

    # 加载第一张图像以确定噪声的形状
    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = Image.open(first_image_path).convert('RGB').resize((720, 480))
    image_array = np.array(first_image)

    # 生成与图像大小相同的随机噪声
    noise = np.random.normal(0, noise_level, image_array.shape).astype(np.int16)

    # 遍历所有图像并添加相同的噪声
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB').resize((720, 480))
        image_array = np.array(image)

        # 添加噪声并裁剪到合法范围
        noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)

        # 保存添加噪声后的图像
        noisy_image_pil = Image.fromarray(noisy_image)
        output_path = os.path.join(output_dir, image_file)
        noisy_image_pil.save(output_path)

    print(f"Processed {len(image_files)} images. Noisy images saved to '{output_dir}'.")

# 示例用法
input_directory = "/home/user/wangxd/diffusers/raw"  # 输入图像文件夹
output_directory = "/home/user/wangxd/diffusers/raw-noise"   # 输出图像文件夹
add_random_noise_to_images(input_directory, output_directory, noise_level=90, seed=42)
