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
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 max pool stride 2
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # conv1 : bottleneck 64, out 256, num 3
        self.layer1 = self._make_layer(in_channel=64, bottleneck_channel=64, out_channel=256, block_num=3, stride=1)

        # conv2 : bottleneck 128, out 512, num 4
        self.layer2 = self._make_layer(in_channel=256, bottleneck_channel=128, out_channel=512, block_num=4, stride=2)

        # conv3 : bottleneck 256, out 1024, num 6
        self.layer3 = self._make_layer(in_channel=512, bottleneck_channel=256, out_channel=1024, block_num=6, stride=2)

        # conv4 : bottleneck 512, out 2048, num 3 => Remove to increse map resolution
        # self.layer4 = self._make_layer(in_channel=1024, bottleneck_channel=512, out_channel=2048, block_num=3, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.w = nn.Conv2d(1024, self.num_classes, 1, bias = False) # conv weight => (num_classes, k, 1, 1)

    def forward(self, x): # (batch_size, 3, 448, 448)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # (batch_size, 1024, 28, 28)
        # x = self.layer4(x) # (batch_size, 2048, 14, 14)
        x = self.conv(x)
        
        GAP = self.w(x)
        S_c = torch.mean(GAP.reshape(GAP.shape[0], self.num_classes, -1), 2)

        return S_c, GAP

    def _make_layer(self, in_channel, bottleneck_channel, out_channel, block_num, stride):

        downsample = None

        if stride != 1 or in_channel != out_channel:
            downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                                              nn.BatchNorm2d(out_channel))

        blocks = []
        blocks.append(Bottleneck_block(in_channel, bottleneck_channel, stride, downsample))
        for i in range(block_num - 1):
            blocks.append(Bottleneck_block(out_channel, bottleneck_channel, 1, None))

        return nn.Sequential(*blocks)



class Bottleneck_block(nn.Module):
    def __init__(self, 
                 in_channel, 
                 bottleneck_channel, 
                 stride,
                 downsample):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, bottleneck_channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.conv2 = nn.Conv2d(bottleneck_channel, bottleneck_channel, 3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        self.conv3 = nn.Conv2d(bottleneck_channel, bottleneck_channel*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channel*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out
