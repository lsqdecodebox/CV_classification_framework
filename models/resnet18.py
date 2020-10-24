import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# 基础模块定义
class Bottleneck(nn.Module):
    #
    expansion = 4
    # 预定义
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 基础conv层和bn层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        # shortcut层
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    # 前向传播
    def forward(self, x):
        # 嵌套式写法节省代码空间
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)



        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # num-class
        # self.num_class = num_class

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=3, bias=False)

        self.bn2 = nn.BatchNorm2d(256)
        #
        self.fc = nn.Linear(256,4)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):

        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        _, _, H1, W1 = p2.size()
        p3 = F.upsample(p3, size=(H1, W1), mode='bilinear')
        p4 = F.upsample(p4, size=(H1, W1), mode='bilinear')
        p5 = F.upsample(p5, size=(H1, W1), mode='bilinear')
        m = torch.cat((p2,p3,p4,p5),1)
        m = F.relu(self.bn2(self.conv2(m)))
        return p2,p3,p4,p5,m


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,320,240)))
    for fm in fms:
        print(fm.size())

# test()

# model = FPN101()
# print(model)

"""
_,_,H1,W1 = p2.size()
        p3 = F.upsample(p3,size=(H1,W1),mode='bilinear')
        p4 = F.upsample(p4, size=(H1, W1), mode='bilinear')
        p5 = F.upsample(p5, size=(H1, W1), mode='bilinear')
        x = p2 + p3 +p4 +p5
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.fc(x)
"""