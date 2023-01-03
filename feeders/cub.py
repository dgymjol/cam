import torch
from torch.utils.data import Dataset
import pickle
import os
import torchvision.transforms as transforms
from PIL import Image

class Feeder(Dataset):
    def __init__(self, data_path, phase):
        # super(Feeder, self).__init__()
        self.num_classes = 200
        self.phase = phase
        
        train_test_id_path = os.path.join(data_path, 'train_test_ids.pkl')
        image_path = os.path.join(data_path, 'data.pkl')
        gt_path = os.path.join(data_path, 'gts.pkl')

        with open(train_test_id_path, 'rb') as fr:
            self.image_ids = pickle.load(fr)

        with open(image_path, 'rb') as fr:
            self.data = pickle.load(fr)

        with open(gt_path, 'rb') as fr:
            self.gt = pickle.load(fr)


        self.transform = transforms.Compose([    transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
                                        ])
        
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
            id = self.image_ids['train'][index]
        else :
            raise Exception("neither train or test")

        image = self.data[id]
        width, height = image.size
        tensor = self.transform(image)
        data = [tensor, [width, height]]
        label = self.gt[id]
        
        return torch.tensor(data), torch.tensor(label), index

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod