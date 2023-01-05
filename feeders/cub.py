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
        
        train_test_id_path = os.path.join(data_path, 'train_test_ids.pkl')
        image_path = os.path.join(data_path, 'data.pkl')
        gt_path = os.path.join(data_path, 'gts.pkl')

        with open(train_test_id_path, 'rb') as fr:
            self.image_ids = pickle.load(fr)

        with open(image_path, 'rb') as fr:
            self.data = pickle.load(fr)

        with open(gt_path, 'rb') as fr:
            self.gt = pickle.load(fr)

        self.transform_org = transforms.Compose([ transforms.Resize((size, size)),
                                              transforms.ToTensor(),
                                            ])

        self.transform_aug = transforms.Compose([ transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                    ])
        
        if aug:
            self.transform_aug.transforms.insert(0, RandAugment(aug_N, aug_M))
        
        self.normalize = transforms.Normalize( mean=[0.4856, 0.4993, 0.4322],
                                               std=[0.2250, 0.2205, 0.2548] )
                            
    def __len__(self):
        if self.phase == 'train':
            return len(self.image_ids['train'])
            
        elif self.phase == 'test':
            return len(self.image_ids['test'])
        else :
            raise Exception("neither train or test")


    def __iter__(self):
        return self
    
    def __getitem__(self, index):

        if self.phase == 'train':
            id = self.image_ids['train'][index]
            
        elif self.phase == 'test':
            id = self.image_ids['test'][index]
        else :
            raise Exception("neither train or test")

        image = self.data[id]
        w, h = image.size

        if len(np.array(image).shape) == 2: # grayscale => no aug
            tensor = self.transform_org(image)
        
        else: # rgb => aug
            tensor = self.transform_aug(image)

        label = self.gt[id]

        if tensor.shape != (3, self.size, self.size): #grayscale
            # print(id) : train 1401
            tensor = tensor.expand(3,self.size, self.size)
        tensor = self.normalize(tensor)
        
        return {"image_data": tensor, "image_size": torch.tensor([w, h]), 
                "gt_cls": torch.tensor(label[0]-1).long(), "gt_box": torch.tensor(label[1])}


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