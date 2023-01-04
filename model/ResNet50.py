import torch
import torch.nn as nn

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

        ## conv1 : 7x7 64 stride 2
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        ## 3x3 max pool stride 2
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)

        residual_blocks = []
        ## conv2 : bottleneck 64, out 256, num 3
        residual_blocks.append(Residual_blocks(in_channel=64, bottleneck_channel=64, out_channel=256, block_num=3, first_layer=True))

        # conv3 : bottleneck 128, out 512, num 4
        residual_blocks.append(Residual_blocks(in_channel=256, bottleneck_channel=128, out_channel=512, block_num=4, first_layer=False))

        ## conv4 : bottleneck 256, out 1024, num 6
        residual_blocks.append(Residual_blocks(in_channel=512, bottleneck_channel=256, out_channel=1024, block_num=6, first_layer=False))

        # conv5 : bottleneck 512, out 2048, num 3
        residual_blocks.append(Residual_blocks(in_channel=1024, bottleneck_channel=512, out_channel=2048, block_num=3, first_layer=False))

        self.rbs = nn.Sequential(*residual_blocks)

        ## avg pooling, fc layer, softmax 
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.mp(self.relu(self.bn(self.conv1(x))))
        x = self.rbs(x)
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)

        return None, x


class Residual_blocks(nn.Module):
    def __init__(self, 
                 in_channel, 
                 bottleneck_channel, 
                 out_channel, 
                 block_num,
                 first_layer):
        super().__init__()

        blocks = []
        if first_layer:
            blocks.append(Residual_block(in_channel, bottleneck_channel, out_channel, 1))
        else:
            blocks.append(Residual_block(in_channel, bottleneck_channel, out_channel, 2))

        for i in range(block_num - 1):
            blocks.append(Residual_block(out_channel, bottleneck_channel, out_channel, 1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Residual_block(nn.Module):
    def __init__(self, 
                 in_channel, 
                 bottleneck_channel, 
                 out_channel, 
                 stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, bottleneck_channel, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(bottleneck_channel, bottleneck_channel, 3,
                               padding=1, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(bottleneck_channel, out_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if stride == 1:
            if in_channel == out_channel:
                self.shortcut = nn.Sequential(nn.Identity(),
                                              nn.BatchNorm2d(out_channel))
            else:
                self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, bias=False),
                                              nn.BatchNorm2d(out_channel))
        elif stride == 2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channel))
        else :
            print(stride)
            assert False, "stride error"
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        sc = self.shortcut(x)

        return out+sc