import torch
import torch.nn as nn
from torch.nn import init

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # conv0 : 7x7 64 stride 2
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 3x3 max pool stride 2
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)

        # conv1 : bottleneck 64, out 256, num 3
        self.layer1 = self._make_layer(in_channel=64, bottleneck_channel=64, out_channel=256, block_num=3, stride=1)

        # conv2 : bottleneck 128, out 512, num 4
        self.layer2 = self._make_layer(in_channel=256, bottleneck_channel=128, out_channel=512, block_num=4, stride=2)

        # conv3 : bottleneck 256, out 1024, num 6
        self.layer3 = self._make_layer(in_channel=512, bottleneck_channel=256, out_channel=1024, block_num=6, stride=2)

        # conv4 : bottleneck 512, out 2048, num 3
        self.layer4 = self._make_layer(in_channel=1024, bottleneck_channel=512, out_channel=2048, block_num=3, stride=2)

        # avg pooling, fc layer, softmax 
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        # self.softmax = nn.Softmax(dim=1)

        self.fc.apply(self._weight_init_kaiming)

    def forward(self, x):
        x = self.mp(self.layer0(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)

        return x, None

    def _make_layer(self, in_channel, bottleneck_channel, out_channel, block_num, stride):

        blocks = []
        blocks.append(Bottleneck_block(in_channel, bottleneck_channel, out_channel, stride))
        for i in range(block_num - 1):
            blocks.append(Bottleneck_block(out_channel, bottleneck_channel, out_channel, 1))

        return nn.Sequential(*blocks)

    def _weight_init_kaiming(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

class Bottleneck_block(nn.Module):
    def __init__(self, 
                 in_channel, 
                 bottleneck_channel, 
                 out_channel, 
                 stride):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channel, bottleneck_channel, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, bottleneck_channel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        if stride == 1:
            if in_channel == out_channel:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                              nn.BatchNorm2d(out_channel))
        elif stride == 2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channel))
        else :
            print(stride)
            assert False, "stride error"

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.convolutions(x) + self.shortcut(x))