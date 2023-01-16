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
import torch.utils.model_zoo as model_zoo

class Processor():

    def __init__(self, arg):
        self.arg = arg

        if not os.path.isdir(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        self.print_log("------------------------")
        self.print_log(str(arg))
        self.print_log("------------------------")
        self.save_arg()
        self.init_seed(self.arg.seed)

        arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
        # if os.path.isdir(arg.model_saved_name):
        #     print('log_dir: ', arg.model_saved_model, 'already exist')
        self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
        self.test_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
        
        
        self.global_step = 0

        self.load_model()
        self.load_data()
        self.load_optimizer()
        self.load_scheduler()
        self.load_loss()

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

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
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
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
        shutil.copy2(inspect.getfile(Feeder), self.arg.work_dir)
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
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.print_log('model : ', Model)
        self.model = Model(**self.arg.model_args)
        
        model_urls = { 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
                       'vgg16' : 'https://download.pytorch.org/models/vgg16-397923af.pth',
                       'vgg16bn' : 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}

                               
        if self.arg.weights == 'Nothing':
            self.print_log("No pretrained weights loaded")
            # raise Exception("No pretrained weights loaded")
        elif self.arg.weights == 'ResNet_ImageNet':
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            self.model.load_state_dict(state_dict, strict=False)
            self.print_log("Successful : transfered weights(ResNet50_imageNet)")
        elif self.arg.weights == 'VGG16_ImageNet':
            state_dict = model_zoo.load_url(model_urls['vgg16'])
            self.model.load_state_dict(state_dict, strict=False)
            self.print_log("Successful : transfered weights(VGG16_imageNet)")
        elif self.arg.weights == 'VGG16bn_ImageNet':
            state_dict = model_zoo.load_url(model_urls['vgg16bn'])
            self.model.load_state_dict(state_dict, strict=False)
            self.print_log("Successful : transfered weights(VGG16bn_imageNet)")
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
                self.print_log(f"Successful : transfered weights ({os.path.join(exp_dir,model_weight_file_name)})")

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=True,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        else:
            raise Exception(f"There is no {self.arg.optimizer}. Add it in load_optimizer().")

    def load_scheduler(self):
        if self.arg.scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor = 0.5, verbose = True)
        elif self.arg.scheduler == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.1)
        else:
            raise Exception(f"There is no {self.arg.scheduler}. Add it in load_scheduler() & step argument")
            
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

        with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
            print(str, file=f)
            
    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('** epoch: {}'.format(epoch + 1))

        loss_value = []
        num_correct = 0
        num_total = 0

        self.train_writer.add_scalar('epoch', epoch, self.global_step)

        for batch_idx, item in enumerate(self.data_loader['train']) :
            self.global_step += 1

            data = item['image_data'].cuda(self.output_device) # (batchsize, 3, 224, 224)
            # image_size = item['image_size'] # (batch_size, 2) : (width, height)
            gt_cls = item['gt_cls'].cuda(self.output_device) # (batchsize,)
            # gt_box = item['gt_box'].cuda(self.output_device) # (batchsize, 4) : (x, y, width, height)
            
            output, _ = self.model(data)
            loss = self.loss(output, gt_cls)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())

            _, pred = torch.max(output, 1)
            num_correct += torch.sum(pred == gt_cls.detach())
            num_total += gt_cls.size(0)
            # self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # self.train_writer.add_scalar('acc', correct.detach().cpu().numpy()*100 / total, self.global_step)
            # print(f'loss : {loss.data.item()}  acc : {num_correct.detach().cpu().numpy()*100 / num_total}')

        train_acc = num_correct.detach().cpu().numpy()*100 / num_total

        if self.arg.scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(np.mean(train_acc))
        else:
            self.scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f"current lr : {current_lr}")
        self.print_log("\t Mean training loss: {:.4f}. Mean training acc: {:.2f}% ".format(np.mean(loss_value), train_acc))
        
        # if save_model:
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '.pt')


    def test(self, epoch):
        self.model.eval()
        loss_value = []

        num_correct = 0
        num_total = 0

        self.test_writer.add_scalar('epoch', epoch, self.global_step)

        for batch_idx, item in enumerate(self.data_loader['test']) :
            self.global_step += 1

            with torch.no_grad():
                data = item['image_data'].cuda(self.output_device) # (batchsize, 3, 224, 224)
                # image_size = item['image_size'] # (batch_size, 2) : (width, height)

                gt_cls = item['gt_cls'].cuda(self.output_device) # (batchsize,)
                # gt_box = item['gt_box'].cuda(self.output_device) # (batchsize, 4) : (x, y, width, height)
                output, _ = self.model(data)
            
                loss = self.loss(output, gt_cls)
                loss_value.append(loss.data.item())

                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == gt_cls.detach())
                num_total += gt_cls.size(0)

                # print(f'loss : {loss.data.item()}  acc : {num_correct.detach().cpu().numpy()*100 / num_total}')

        test_acc = num_correct.detach().cpu().numpy()*100/num_total

        # if test_acc > self.best_acc:
        #     self.best_acc = test_acc
        #     self.best_acc_epoch = epoch + 1
        #     state_dict = self.model.state_dict()
        #     weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        #     torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '.pt')

        self.print_log("\t Mean test loss: {:.4f}. Mean test acc: {:.2f}%.".format(np.mean(loss_value), test_acc))

        # self.test_writer.add_scalar('lr', self.lr, self.global_step)
        # self.test_writer.add_scalar('top1', test_acc, self.global_step)


    def start(self):
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            save_model = (((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch
            self.train(epoch, save_model=save_model)
            self.test(epoch)


        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Best accuracy: {self.best_acc}')
        self.print_log(f'Epoch number: {self.best_acc_epoch}')
        self.print_log(f'Model name: {self.arg.work_dir}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
        self.print_log(f'seed: {self.arg.seed}')