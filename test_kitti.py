import sys

sys.path.append('/home/user/wangxd/diffusers/deq-flow/code.v.2.0/core')

from deq_flow import DEQFlow
from deq.arg_utils import add_deq_args
from utils.utils import InputPadder

import cv2
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
from PIL import Image

from utils import flow_viz, frame_utils
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 

    parser.add_argument('--total_run', type=int, default=1, help="total number of runs")
    parser.add_argument('--start_run', type=int, default=1, help="begin from the given number of runs")
    parser.add_argument('--restore_name', help="restore experiment name")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--tiny', action='store_true', help='use a tiny model for ablation study')
    parser.add_argument('--large', action='store_true', help='use a large model')
    parser.add_argument('--huge', action='store_true', help='use a huge model')
    parser.add_argument('--gigantic', action='store_true', help='use a gigantic model')
    parser.add_argument('--old_version', action='store_true', help='use the old design for flow head')

    parser.add_argument('--restore_ckpt', help="restore checkpoint for val/test/viz")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=5000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0, help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--active_bn', action='store_true')
    parser.add_argument('--all_grad', action='store_true', help="Remove the gradient mask within DEQ func.")

    # Add args for utilizing DEQ
    add_deq_args(parser)
    args = parser.parse_args()
    
    args.stage = "things"
    args.gpus = 0
    args.wnorm = True
    args.f_thres = 40
    args.f_solver =  "naive_solver"
    args.huge = True
    args.eval_factor = 3.0
    


    device = "cuda:2"

    model = DEQFlow(args)
    state_dicts = torch.load("/home/user/wangxd/diffusers/deq-flow/code.v.2.0/deq-flow-H-things-test-1x.pth", map_location="cpu")
    new_state_dict = {}
    for k, v in state_dicts.items():
        new_k = k[7:]
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()


    transform = transforms.Compose(
        [
            transforms.Resize((480, 720)),
            transforms.PILToTensor()
        ]
    )

    image1 = Image.open("/home/user/wangxd/diffusers/origin_imgs/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151681162404.jpg")
    image2 = Image.open("/home/user/wangxd/diffusers/origin_imgs/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151682162404.jpg")

    image1 = transform(image1)
    image2 = transform(image2)
    
    # print(image1.max(), image1.min()) # [0, 255]
    
    with torch.no_grad():
        padder = InputPadder(image1.shape, mode='kitti')

        image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))

        import time
        t_s = time.time()

        _, flow_pr, _ = model(image1, image2)
        
        t_d = time.time()
        
        print(t_d-t_s)
        
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu()#.numpy()
        # import pdb; pdb.set_trace()

        output_filename = os.path.join("/home/user/wangxd/diffusers/deq-flow/code.v.2.0", "test_gray_9.jpg")

        # os.makedirs(output_filename, exist_ok=True)

        # visualizaion
        img_flow = flow_viz.flow_to_image(flow)
        
        # img_flow = cv2.resize(img_flow, (90, 60))
        # img_flow = cv2.cvtColor(img_flow, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_filename, img_flow, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
