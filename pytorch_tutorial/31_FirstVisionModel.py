import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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


### 3. First Vision Model (baseline): model_0
'''
    When starting to build a series of machine learning modelling experiments, 
    it's best practice to start with a baseline model.

    A baseline model is a simple model you will try and improve upon with subsquent to models/experiments.

    In other words, start simply and add complexity when necessary.  
'''

# Creat a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x) # perform forward pass

# Print out what happen
print(f"\nShape before flattening: {x.shape} -> [color_channels, height, width]\n"
        f"Shape after  flattening: {output.shape} -> [color_channels, height*width]\n")


class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features= hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features= output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)

# Setup model with input params
model_0 = FashionMNISTModelV0(
    input_shape=28*28,
    hidden_units=10,
    output_shape=len(class_names)
).to("cpu")

print(model_0)

dummy_x = torch.rand([1,1,28,28])
dummy_result = model_0(dummy_x)
print(f"test dummy_x:\n {dummy_result}\n"
      f"shape of the result: {dummy_result.shape}\n")

print(f"\n\nmodel_0 state_dict: \n{model_0.state_dict()}\n")







debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_46_03
# 14_51_41 (2026-04-14)
# 15_21_15 (2026-04-15)
# 15_35_55 (2026-04-20)