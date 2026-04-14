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
# See the first training example
image, label = train_data[0]
# Check class names
class_names = train_data.classes
print(f"class names:\n{class_names}\n")
# Check data length
print(f"length of train_data:\n{len(train_data)}\nlength of test_data\n{len(test_data)}\n")
# Check class idx
class_to_idx = train_data.class_to_idx
print(f"Check class idx:\n{class_to_idx}\n")
# Check shape
print(f"Image shape:\n{image.shape} -> [color_channels, height, width]\n")
print(f"Image label:\n{class_names[label]}\n")
# Why the color channel is 1?
'''
    0->black, 1->full color [here is white], between 0 and 1 -> grey
'''




debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_46_03
# 14_51_41 too busy on work and driving