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
        
def get_parser():
    parser = argparse.ArgumentParser(
        description='Visualization of CAM')
    
    parser.add_argument(
            '--weights',
            default='work_dir/cub/resnet50_cam:60')

    parser.add_argument(
            '--test-image-path',
            # default='data/CUB_200_2011/images/015.Lazuli_Bunting/Lazuli_Bunting_0020_14837.jpg')    
            # default='data/CUB_200_2011/images/015.Lazuli_Bunting/Lazuli_Bunting_0084_14815.jpg')
            default='data/CUB_200_2011/images/184.Louisiana_Waterthrush/Louisiana_Waterthrush_0059_177449.jpg')
    return parser

init_seed(1)
parser = get_parser()
arg = parser.parse_args()

exp_dir, epoch = arg.weights.split(':')
if not os.path.exists(exp_dir):
    raise Exception(f"{exp_dir} : No such file or directory")
else:
    model_weight_file_name = ''
    for run_file in os.listdir(exp_dir) :
        if f"runs-{epoch}" in run_file  and '.pt' in run_file:
            model_weight_file_name = run_file
    if model_weight_file_name == '':
        raise Exception(f'that epoch {epoch} weight file doesnt exist')

    weights = torch.load(os.path.join(exp_dir,model_weight_file_name))
    model = ResNet50_cam(num_classes=200)
    model.load_state_dict(weights, strict=False)
    print("pretrained weight is loaded")

test_image_path = arg.test_image_path

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
print(conf)
print(gaps)
label = torch.argmax(conf[0]) # 55
gap = gaps[0][label] # (24, 32)

# map_image = iresize(t2i(gap))
# map_image.save('map.jpg',"JPEG")

gap = gap - torch.min(gap)
gap = gap / torch.max(gap)

gap = gap.detach().cpu().numpy()
gap_image = np.uint8(255*gap)
print(gap_image)
heatmap = cv2.applyColorMap(cv2.resize(gap_image,(w, h)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + test_image_cv * 0.5

print("ground truth label : ", gt_cls)
print("predicited label : ", int(label.data))

plag = True
if int(label.data) != int(gt_cls) - 1:
    plag = False

cv2.imwrite(f'cam_for_class_{img_name}_{plag}.jpg', heatmap)
cv2.imwrite(f'cam_for_class_{img_name}_{plag}_compare.jpg', result)


cam_image = cv2.resize(gap_image,(w, h))
threshold = np.max(cam_image) * 0.4

    # threshold map 
_, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_BINARY)
    # or (this one : Otsu's binarization method 적용한 것)
# _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_OTSU)


"""
If you want to generate threshold map : below
"""
thresh_map_rgb = np.zeros_like(test_image_cv)

thresh_map_rgb[:, :, 0] = thresh_map
thresh_map_rgb[:, :, 1] = thresh_map
thresh_map_rgb[:, :, 2] = thresh_map

result = thresh_map_rgb * 0.3 + test_image_cv * 0.5

cv2.imwrite(f'th_{img_name}.jpg', thresh_map)
cv2.imwrite(f'th_{img_name}compare.jpg', result)

"""
generating bounding box and evaluate IOU
"""

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)

largest_connected_component_idx = np.argmax(stats[1:, -1]) + 1 # background is most
bbox = stats[largest_connected_component_idx][:-1] #(x, y, width, height)