from torchvision.models import vgg16
import torch
from torch.autograd import Variable
import torch.nn as nn

model = vgg16()
print(model)