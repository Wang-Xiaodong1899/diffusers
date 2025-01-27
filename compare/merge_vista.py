import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import cv2
import os
import argparse
from PIL import Image
from tqdm import tqdm

from diffusers.utils import load_image, export_to_video

def merge_videos(fps):
    """
    Extract frames from an MP4 file at a specified FPS and save them to an output directory.

    Args:
        mp4_path (str): Path to the MP4 file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int): Frames per second to extract.
    """
    # Create the output directory if it doesn't exist
    output_dir = "9-0003" # TODO
    
    os.makedirs(output_dir, exist_ok=True)
    mp4_paths = ['9-0003-vista.mp4', '9-0003-vista-continue.mp4'] # TODO
    frames = []
    
    for (idx, mp4_path) in tqdm(enumerate(mp4_paths)):

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
                # frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
                # cv2.imwrite(frame_filename, frame)
                saved_count += 1
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                if idx >0 and frame_count in [0, 1, 2]:
                    pass
                else:
                    frames.append(pil_image)

            frame_count += 1

        video.release()
    # print(f"Extracted {saved_count} frames to {output_dir}")
    # export to video
    export_to_video(frames, os.path.join(output_dir, f"vista_long.mp4"), fps=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 file.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second to extract (default: 1).")

    args = parser.parse_args()

    merge_videos(args.fps)