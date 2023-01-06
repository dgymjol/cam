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


def init_seed(seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

init_seed(1)
model = ResNet50_cam(num_classes=200)

exp_dir = 'work_dir/cub/resnet50_cam'
epoch = 33
for run_file in os.listdir(exp_dir) :
    if f"runs-{epoch}" in run_file  and '.pt' in run_file:
        model_weight_file_name = run_file
weights = torch.load(os.path.join(exp_dir,model_weight_file_name))

model.load_state_dict(weights, strict=False)

test_image_path = './data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0040_796066.jpg'
img_name = test_image_path.split('/')[-1]
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

plag = True
if int(label.data) != int(gt_cls) - 1:
    plag = False
print("ground truth label : ", gt_cls)
print("predicited label : ", int(label.data))

cam_image = cv2.resize(gap_image,(w, h))
threshold = np.max(cam_image) * 0.8
thresh_value, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_BINARY)


thresh_map_rgb = np.zeros_like(test_image_cv)

thresh_map_rgb[:, :, 0] = thresh_map
thresh_map_rgb[:, :, 1] = thresh_map
thresh_map_rgb[:, :, 2] = thresh_map

result = thresh_map_rgb * 0.3 + test_image_cv * 0.5

cv2.imwrite('th.jpg', thresh_map)
cv2.imwrite('th_compare.jpg', result)






