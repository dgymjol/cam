import os
import numpy as np
import random
import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tensorboardX import SummaryWriter
import time
import argparse
import yaml
import pdb
import inspect
import shutil
from collections import OrderedDict
import argparse
import yaml
from processor import Processor
import cv2

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Computer Vision Task Training Processor')

    parser.add_argument(
        '--results-dir',
        default='./results_dir/temp')

    parser.add_argument(
        '--config',
        default='./config/cub/resnet50_cam_eval.yaml')


    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='the interval for printing messages (#iteration)')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')

    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=100,
        help='the number of classes')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='Nothing',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss', help='type of optimizer')

    return parser
class Processor():

    def __init__(self, arg):
        self.arg = arg

        if not os.path.isdir(self.arg.results_dir):
            os.makedirs(self.arg.results_dir)

        self.print_log("------------------------")
        self.print_log(str(arg))
        self.print_log("------------------------")
        self.save_arg()
        self.init_seed(self.arg.seed)

        self.load_model()
        self.load_data()
        self.load_loss()

        self.model = self.model.cuda(self.output_device)
        
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids = self.arg.device, output_device=self.output_device)

    def import_class(self, import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError(f'Class {class_str} cannot be found')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.results_dir):
            os.makedirs(self.arg.results_dir)
        with open('{}/config.yaml'.format(self.arg.results_dir), 'w') as f:
            f.write(f"# commend line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def init_seed(self, seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self):
        
        self.data_loader = dict()

        Feeder = self.import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(Feeder), self.arg.results_dir)
        self.data_loader['train'] = DataLoader(
                                    dataset=Feeder(**self.arg.train_feeder_args),
                                    batch_size=self.arg.batch_size,
                                    shuffle=True,
                                    num_workers=self.arg.num_worker)
        self.data_loader['test'] =  DataLoader(
                                    dataset=Feeder(**self.arg.test_feeder_args),
                                    batch_size=self.arg.test_batch_size,
                                    shuffle=False,
                                    num_workers=self.arg.num_worker)

    def load_model(self):
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = self.import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.results_dir)
        self.print_log('model : ', Model)
        self.model = Model(**self.arg.model_args)
        
        if self.arg.weights == 'Nothing':
            self.print_log("No pretrained weights loaded")
            # raise Exception("No pretrained weights loaded")
        else:
            exp_dir, epoch = self.arg.weights.split(':')
            if not os.path.exists(exp_dir):
                self.print_log(f"Error : the dir doesnt exist {exp_dir}")
                raise Exception(f"the dir doesnt exist {exp_dir}")
            else:
                model_weight_file_name = ''
                for run_file in os.listdir(exp_dir) :
                    if f"runs-{epoch}.pt" in run_file:
                        model_weight_file_name = run_file
                if model_weight_file_name == '':
                    self.print_log(f'Error : that epoch{epoch} weight file doesnt exist')
                    raise Exception(f'that epoch{epoch} weight file doesnt exist')

                weights = torch.load(os.path.join(exp_dir,model_weight_file_name))
                self.model.load_state_dict(weights, strict=False)
                self.print_log(f"Successful : transfered weights ({os.path.join(exp_dir,model_weight_file_name)}, epoch {epoch})")

    def load_loss(self):
        if self.arg.loss == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        else:
            raise Exception(f"There is no {self.arg.loss}. Add it in load_loss().")     

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)

        with open('{}/result.txt'.format(self.arg.results_dir), 'a') as f:
            print(str, file=f)

    def IoU(self, box1, box2):
        '''
            box1 : (x, y, w, h)
            box2 : (x, y, w, h)
        '''

        # intersection x1, y1, x2, y2

        # box = (x1, y1, x2, y2)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2]) # x + w
        y2 = min(box1[1] + box1[3], box2[1] + box2[3]) # y + h

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        iou = inter / (box1_area + box2_area - inter + 1e-7)
        return iou

    def eval_train(self):
        self.model.eval()
        loss_value = []

        num_correct = 0
        num_total = 0
        sum_iou = 0

        for batch_idx, item in enumerate(self.data_loader['train']) :
            with torch.no_grad():
                data = item['image_data'].cuda(self.output_device) # (1, 3, 224, 224)
                w, h = item['image_size'][0] # (1, 2) : (width, height)
                gt_cls = item['gt_cls'].cuda(self.output_device) # (1,)
                gt_box = item['gt_box'][0].numpy() # (1, 4) : (x, y, width, height)
                conf, gaps = self.model(data)

                # classification
                _, pred_label = torch.max(conf, 1)
                loss = self.loss(conf, gt_cls)
                loss_value.append(loss.data.item())
                if pred_label == gt_cls.detach() :
                    num_correct += 1
                num_total += 1

                # localization
                gap = gaps[0][pred_label][0] # (14, 14)
                gap = gap - torch.min(gap)
                gap = gap / torch.max(gap)
                gap = gap.detach().cpu().numpy()
                gap_image = np.uint8(255*gap)
                cam_image = cv2.resize(gap_image,(int(w), int(h)))
                threshold = np.max(cam_image) * 0.20
                _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_BINARY)
                # _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_OTSU)
                cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)

                largest_connected_component_idx = np.argmax(stats[1:, -1]) + 1 # background is most
                pred_box = stats[largest_connected_component_idx][:-1] #(x, y, width, height)

                iou = self.IoU(gt_box, pred_box)
                sum_iou += iou
        
        train_acc = num_correct*100/num_total
        train_mIoU = sum_iou / num_total

        self.print_log("\t Mean train loss: {:.4f}. Mean train acc: {:.2f}%. mIoU: {:4f}".format(np.mean(loss_value), train_acc, train_mIoU))

    def eval_test(self):
        self.model.eval()
        loss_value = []

        num_top1_cls = 0
        num_top1_loc = 0
        num_gt_know_loc = 0

        num_total = 0
        sum_iou = 0

        for batch_idx, item in enumerate(self.data_loader['test']) :
            with torch.no_grad():
                data = item['image_data'].cuda(self.output_device) # (1, 3, 224, 224)
                w, h = item['image_size'][0] # (1, 2) : (width, height)
                gt_cls = item['gt_cls'].cuda(self.output_device) # (1,)
                gt_box = item['gt_box'][0].numpy() # (1, 4) : (x, y, width, height)
                conf, gaps = self.model(data)

                cls = True
                # classification
                _, pred_label = torch.max(conf, 1)
                loss = self.loss(conf, gt_cls)
                loss_value.append(loss.data.item())
                if pred_label == gt_cls.detach() :
                    num_top1_cls += 1
                else:
                    cls = False

                num_total += 1

                # localization
                gap = gaps[0][pred_label][0] # (14, 14)
                gap_min, gap_max = torch.min(gap), torch.max(gap)
                gap = (gap-gap_min) / (gap_max - gap_min)
                gap = gap.detach().cpu().numpy()
                gap_image = np.uint8(255*gap)
                cam_image = cv2.resize(gap_image,(int(w), int(h)))
                threshold = np.max(cam_image) * 0.20
                _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_BINARY)
                # _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_OTSU)

                cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)

                largest_connected_component_idx = np.argmax(stats[1:, -1]) + 1 # background is most
                pred_box = stats[largest_connected_component_idx][:-1] #(x, y, width, height)

                iou = self.IoU(gt_box, pred_box)
                sum_iou += iou
                
                bbox = False
                if iou >= 0.5:
                    bbox = True
                    num_gt_know_loc += 1

                if bbox and cls:
                    num_top1_loc += 1
                

        test_top1_cls = num_top1_cls * 100 / num_total 
        test_top1_loc = num_top1_loc * 100 / num_total 
        test_gt_know_loc = num_gt_know_loc * 100 / num_total 

        test_mIoU = sum_iou / num_total

        self.print_log("\t Mean test top1_cls: {:.2f}%. Mean test top1_loc: {:.2f}%. Mean test gt_known_loc: {:.2f}%. mIoU: {:4f}".format(test_top1_cls, test_top1_loc, test_gt_know_loc, test_mIoU))

    def start(self):
        self.eval_test()
        # self.eval_train()

if __name__ == '__main__':
    parser = get_parser()

    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()

        for k in default_arg.keys():
            if k not in key:
                print(f'Wrong arg : {k}')
                assert(k in key)

        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    processor = Processor(arg)
    processor.start()