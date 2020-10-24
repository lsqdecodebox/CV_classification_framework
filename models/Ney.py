import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,num_class):
        super(Net, self).__init__()
        self.basic = vgg16().features
        self.dilated_1 = nn.Conv2d(512, 1024, kernel_size=(3,3), padding=(1,1))
        self.dilated_2 = nn.Conv2d(512, 1024, kernel_size=(3,3), padding=(3,3),dilation=3)
        self.dilated_3 = nn.Conv2d(512, 1024, kernel_size=(3,3), padding=(6,6),dilation=6)
        self.dilated_4 = nn.Conv2d(512, 1024, kernel_size=(3,3), padding=(9,9),dilation=9)
        self.conv1 = nn.Conv2d(1024,512,kernel_size=(1,1),padding=(1,1))
        self.conv2 = nn.Conv2d(2048,512,kernel_size=(1,1),padding=(1,1))
        self.conv3 = nn.Conv2d(512,128,kernel_size=(1,1),padding=(1,1))
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding=0)
        # self.fc = nn.Conv2d(128,4,kernel_size=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(61952,1024)
        self.fc2 = nn.Linear(1024,num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.basic(x)
        x_1 = self.pool1(F.relu(self.conv1(self.dilated_1(x))))
        x_2 = self.pool1(F.relu(self.conv1(self.dilated_2(x))))
        x_3 = self.pool1(F.relu(self.conv1(self.dilated_3(x))))
        x_4 = self.pool1(F.relu(self.conv1(self.dilated_4(x))))
        x = torch.cat((x_1,x_2,x_3,x_4),1)
        # self.cat = torch.cat((x_1,x_2,x_3,x_4))
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = x.view(-1,61952)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x






