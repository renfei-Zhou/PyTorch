import pandas as pd
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


'''
    train data, involving torchvision, transform, DataLoader
'''


dataset = torchvision.datasets.CIFAR10("../data/bees", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)    
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output
    

mymodel = MyModel()


writer = SummaryWriter("../logs_maxpool")

for step,data in enumerate(dataloader):
    imgs, tatgets = data
    writer.add_images("input", imgs, step)
    output = mymodel(imgs)
    writer.add_images("output", output, step)

writer.close()


debug=1