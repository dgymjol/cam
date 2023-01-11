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


from feeders.cub import Feeder

from model.ResNet50 import ResNet50
from model.resnet import ResNet

from model.ResNet50_cam_m import ResNet50_cam
model = ResNet50_cam(num_classes=200)







# bad_image = cv2.imread("data_bad.jpg")
# good_image = cv2.imread("data_good.jpg")

# print(np.allclose(bad_image, good_image ))

model_good = torch.load('model_good.pth')
model_bad = torch.load('model_bad.pth')

for (a_key, a_value), (b_key, b_value) in zip(model_good.items(), model_bad.items()):
    if a_key != 'module.'+b_key:
        print("diff key")
        print(a_key, b_key)
        break
    if not np.allclose(a_value.cpu().numpy(), b_value.cpu().numpy()):
        print("diff value")
        break


exit()

# with open('plz_resnet.txt', 'a') as f:
#     print(model, file=f)

# model = ResNet()

# with open('plz_resnet_git.txt', 'a') as f:
#     print(model, file=f)
# x = torch.randn(128, 3, 224, 224)

# print(model(x)[1].shape)

# exit()
# ds = Feeder()
# breakpoint()
# data_loader = DataLoader(
#                                     dataset=ds,
#                                     batch_size=16,
#                                     shuffle=False,
#                                     num_workers=8)
# a = 0
# for batch_idx, item in enumerate(data_loader) :
#     a+=1

# exit()
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return 

    
exp_dir = './work_dir/cbu/resnet50_cam_0.0001_Adam'
model_weight_file_name = ''
epoch = 1
for run_file in os.listdir(exp_dir) :
    if f"runs-{epoch}" in run_file  and '.pt' in run_file:
        model_weight_file_name = run_file
weights = torch.load(os.path.join(exp_dir,model_weight_file_name))

model = ResNet50_cam(in_channel=3, num_classes=200)

model.load_state_dict(weights)

data_loader =  DataLoader(  dataset=Feeder('./data/CUB_200_2011/', 'test'),
                            batch_size=2,
                            shuffle=False,
                            num_workers=8)        

for batch_idx, item in enumerate(data_loader) :
    break

output = model(item['image_data'])
up = nn.Upsample(size=(500,335))

cam = up(output[0][0].squeeze(0).detach().cpu().numpy())

heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
cv2.imwrite('CAM.jpg', heatmap)

img = cv2.imread('test.jpg')

exit()
import numpy as np
import os
import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pdb

from tensorboardX import SummaryWriter
import time
import argparse

from model import ResNet50
from sklearn.metrics import top_k_accuracy_score
from model.ResNet50_cam import ResNet50_cam
from model.ResNet50 import ResNet50
from model.resnet import resnet50




def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_log(self, str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)
    # if self.args.print_log:
    #     with open('{}/log.txt'.format(self.args.work_dir), 'a') as f:
    #         print(str, file=f)

    with open('log.txt', 'a') as f:
        print(str, file=f)


parser = argparse.ArgumentParser(description='training condition')
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--lr', type = float, default = 0.0001)

args = parser.parse_args()

init_seed(1)

transform = transforms.Compose([    transforms.Resize((32*8, 32*8)),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                    transforms.RandomHorizontalFlip(p = 1),
                                    transforms.ToTensor()
                                ])

train_dataset = datasets.CIFAR100(root='./data',
                                  train=True,
                                  download=True,
                                  transform=transform)

test_dataset = datasets.CIFAR100(root='./data',
                                  train=False,
                                  download=True,
                                  transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=True)

use_cuda = torch.cuda.is_available()                   # check if GPU exists
DEVICE = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
print(f"DEVICE : {DEVICE}")

# model = resnet50().to(DEVICE)
model = ResNet50(in_channel=3, num_classes = 100).to(DEVICE)
# model = ResNet50_cam(in_channel=3, num_classes = 100).to(DEVICE)
# from torchvision import models
# import torch

# resnet50 = models.resnet18(pretrained=False)

with open('./resnet.txt', 'a') as f:
    print(model, file=f)
    # print(resnet50, file=f)

exit()

optimizer = optim.Adam(model.parameters(), lr=args.lr, )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor = 0.5, verbose = True)
lossfn = nn.CrossEntropyLoss().to(DEVICE)

# train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')

# loss_value = []
# acc_value = []
# f1_value = []


loss_value = []
labels = []
pred_scores = []

for batch_idx, (data, label) in enumerate(train_loader) :

    if batch_idx == 10: 
        break

    # optimizer.zero_grad()
    with torch.no_grad():
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        # label = F.one_hot(label)
        # data = data.to(DEVICE)
        # label = label.to(DEVICE)
        output = model(data)
        print(output.shape)
        exit()
        # breakpoint()
        # loss = lossfn(output, label)
        # print(loss)
        # exit()
        # loss_value.append(loss.data.item())
        # pred_scores.append(output.cpu().numpy())
        # labels.append(label.cpu().numpy())

# labels = np.concatenate(labels)
# pred_scores = np.concatenate(pred_scores)

# breakpoint()

# top1_acc = top_k_accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy(), k=1, labels=np.arange(self.arg.num_class))
# top5_acc = top_k_accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy(), k=5, labels=np.arange(self.arg.num_class))
# top1_value.append(top1_acc)
# top5_value.append(top5_acc)       
    
# top_k_accuracy_score(labels, pred_scores, k=1, labels=np.arange(100))
    # print(data.shape)
    # print(label.shape)
    # print(label[1:10])
    # print(model(data).shape)
    # break  
    # optimizer.zero_grad()
    # output = model(data)
    # print(label[1])
    # print(output[1,:])
    # print(loss(output, label))
    # print(top_k_accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy(), k=2, labels=np.arange(100)))
    # loss.backward()
    # optimizer.step()

    # n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

    # break
    # loss_value.append(loss(output, label))
    # train_writer.add_scalar('epoch', loss, batch_idx)
    
    
    # with torch.no_grad():
