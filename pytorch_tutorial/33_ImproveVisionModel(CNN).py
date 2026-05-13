import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
from my_helper_functions import accuracy_fn, print_train_time, train_step, test_step, eval_model
# Necessary data -------------------------------------
torch.manual_seed(42)
# Setup training data
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)
# class name
class_names = train_data.classes
# batch size
BATCH_SIZE = 32
# Turn dataset into iterables (batches)
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
# Setup iterable
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
# Necessary data ends here -------------------------------------



### Model 2: Buildiing a Convolutional Neural Network (CNN)
'''
    CNN's are also known ConvNets.
    CNN's are known for their capability to find patterns in visual data.
    To find what's happening inside a CNN: https://poloclub.github.io/cnn-explainer/
'''
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    '''
    Model architecture that replicates the TinyVGG
    model from CNN explainer website.
    '''
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create a conv layer - http://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # 3x3 is the most common kernel size
                      stride=1, # how many pixels the kernel moves at a time
                      padding=1), # Values we can set ourselves in our NN's are called HYPERparameters
            nn.Relu(),
            nn.Conv2d(inchannels=hidden_units,
                      outchannels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.nn_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.linear(in_features=hidden_units*0,
                      out_features=output_shape)
        )

    def forward(self, x):








debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_51_41 (2026-04-14)
# 15_21_15 (2026-04-15)
# 15_35_55 (2026-04-20)
# 16_06_00 (2026-04-27)
# 16_25_08 (2026-04-29)
# 17_09_23 (2026-05-05)
# test: now using macbook for next coding
# 17_37_23 (2026-05-13)