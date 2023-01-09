import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import os
import torchvision.transforms as transforms
from RandAugment import RandAugment
from PIL import Image


class Feeder(Dataset):
    def __init__(self, data_path='data/CUB_200_2011', phase='train', size=448, aug=True, aug_N=2, aug_M=9):
        # super(Feeder, self).__init__()
        self.num_classes = 200
        self.phase = phase
        self.size = size
        
        self.train_data_path = os.path.join(data_path, 'train_data.pkl')
        self.test_data_path = os.path.join(data_path, 'test_data.pkl')

        if phase == 'train':
            with open(self.train_data_path, 'rb') as fr:
                self.data = pickle.load(fr)
        else :
            with open(self.test_data_path, 'rb') as fr:
                self.data = pickle.load(fr)

        self.transform_eval = transforms.Compose([ transforms.Resize(int(size/0.875)),
                                                   transforms.CenterCrop(size),
                                                   transforms.ToTensor(),
                                                 ])     

        self.normalize = transforms.Normalize( mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
                            
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self
    
    def __getitem__(self, index):

        id, pil_image, label, bbox = self.data[index]

        tensor = self.transform_eval(pil_image)
        if tensor.shape != (3, self.size, self.size): #grayscale
            tensor = tensor.expand(3,self.size, self.size)
        tensor = self.normalize(tensor)
        
        return {"image_data": tensor, "image_id": torch.tensor(id),
                "gt_cls": torch.tensor(int(label)-1).long(), "gt_box": torch.tensor(bbox)}


# def find_mean_std():
#     # https://eehoeskrap.tistory.com/463
#     data_loader = DataLoader( dataset=Feeder('../data/CUB_200_2011'),
#                               batch_size=128,
#                               shuffle=False,
#                               num_workers=8)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for batch_idx, item in enumerate(data_loader) :
#         with torch.no_grad():
#             data = item['image_data']# (batchsize, 3, 224, 224)
#             # breakpoint()
#             for i in range(3):
#                 try: 
#                     mean[i] += data[:, i, :, :].mean()
#                     std[i] += data[:, i, :, :].std()
#                 except:
#                     breakpoint()
#     mean.div_(len(data_loader))
#     std.div_(len(data_loader))
#     #tensor([0.4856, 0.4993, 0.4322]) : mean
#     #tensor([0.2250, 0.2205, 0.2548]) : std
#     return mean, std

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod