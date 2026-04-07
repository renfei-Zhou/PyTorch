### 0. Computer vision libaries in PyTorch
'''
    * torchvision - base domain library for PyTorch computer vision
    * torchvision.datasets - get datasets and data loading functions for computer vision
    * torchvision.models - get pretrained computer vision models that you can leverage for your own problems
    * torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
    * torch.utils.data.Dataset - base dataset class for PyTorch
    * torch.utils.data.DataLoader - creates a Python iterable over a dataset
'''
# import pytorch
import torch
from torch import nn
# import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
# import matplotlib for visualization
import matplotlib.pyplot as plt

### 1. Getting a dataset
'''
    The dataset we'll be using is FashionMNIST from torchvision.dataset
'''
# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # Where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=torchvision.transforms.ToTensor(), # how do we want to transfer the data? 
    target_transform=None # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)







debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_46_03