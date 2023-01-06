import os
import numpy as np
import random
import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import time
import argparse
import yaml
import pdb
import inspect
import shutil
from collections import OrderedDict
from model.ResNet50_cam import ResNet50_cam
from feeders.cub import Feeder
import cv2
from PIL import Image
import torchvision.transforms as transforms

from feeders.cub import Feeder

from model.ResNet50 import ResNet50
from model.ResNet50_cam import ResNet50_cam

model = ResNet50_cam(num_classes=200)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Visualization of CAM')
    
    parser.add_argument(
            '--exp-dir',
            default='./work_dir/cub/resnet50')

    parser.add_argument(
            '--epoch',
            type=int,
            default=62)
    parser.add_argument(
            '--test-image-path',
            default='./data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0040_796066.jpg')    
    return parser

def visualize_cam(arg):
    exp_dir = arg.exp_dir
    epoch = arg.epoch
    test_image_path = arg.test_image_path

    for run_file in os.listdir(exp_dir) :
        if f"runs-{epoch}" in run_file  and '.pt' in run_file:
            model_weight_file_name = run_file
    weights = torch.load(os.path.join(exp_dir,model_weight_file_name))

    model.load_state_dict(weights, strict=False)

    img_name = test_image_path.split('/')[-1].split('.')[0]
    gt_cls = int(test_image_path.split('/')[-2].split('.')[0]) - 0

    test_image = Image.open(test_image_path)
    test_image_cv = cv2.imread(test_image_path)

    w, h = test_image.size

    i2t = transforms.ToTensor()
    t2i = transforms.ToPILImage()
    iresize = transforms.Resize((h, w))

    t = i2t(test_image) # (3, 324, 500)
    conf, gaps = model(t.unsqueeze(0)) # [1, 200] [1, 200, 24 ,32]
    label = torch.argmax(conf[0]) # 55
    gap = gaps[0][label] # (24, 32)

    # map_image = iresize(t2i(gap))
    # map_image.save('map.jpg',"JPEG")

    gap = gap - torch.min(gap)
    gap = gap / torch.max(gap)

    gap = gap.detach().cpu().numpy()
    gap_image = np.uint8(255*gap)
    heatmap = cv2.applyColorMap(cv2.resize(gap_image,(w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + test_image_cv * 0.5

    print("ground truth label : ", gt_cls)
    print("predicited label : ", int(label.data))

    plag = True
    if int(label.data) != int(gt_cls) - 1:
        plag = False

    cv2.imwrite(f'cam_for_class_{img_name}_{plag}.jpg', heatmap)
    cv2.imwrite(f'cam_for_class_{img_name}_{plag}_compare.jpg', result)

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    visualize_cam(p)