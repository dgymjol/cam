import torch
import torch.nn as nn

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
    
class ResNet50_cam(nn.Module):
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

        # conv4 : bottleneck 512, out 2048, num 3 => Remove to increse map resolution
        # self.layer4 = self._make_layer(in_channel=1024, bottleneck_channel=512, out_channel=2048, block_num=3, stride=2)

        self.final_conv = nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False)
        self.w = nn.Conv2d(1024, self.num_classes, 1, bias = False) # conv weight => (num_classes, k, 1, 1)

    def forward(self, x):
        x = self.mp(self.layer0(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        GAP = self.w(x)
        S_c = torch.sum(GAP.reshape(GAP.shape[0], self.num_classes, -1), 2)

        return S_c, GAP

    def _make_layer(self, in_channel, bottleneck_channel, out_channel, block_num, stride):

        blocks = []
        blocks.append(Bottleneck_block(in_channel, bottleneck_channel, out_channel, stride))
        for i in range(block_num - 1):
            blocks.append(Bottleneck_block(out_channel, bottleneck_channel, out_channel, 1))

        return nn.Sequential(*blocks)



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