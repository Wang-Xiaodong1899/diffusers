import numpy as np
from PIL import Image
import cv2
import json

def image2pil(filename):
    return Image.open(filename).convert('RGB')

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr

def mp4toarr(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_uint8 = frame.astype(np.uint8)
        frames.append(frame_uint8)
    cap.release()
    return np.array(frames)

def json2data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def data2json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)