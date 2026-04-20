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

### 1.1 Check the proporties of the train_data
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

### 1.2 Visualing our data
import matplotlib.pyplot as plt
# image, label = train_data[0]
# print(f"\nImage shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# # Plot more images
# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()

'''
    Question: 
        Could the images be modelled with pure linear lines?
        Or non-linearities need?
'''


### 2. Prepare DataLoader
'''
    Right now, our data is in the form of PyTorch Datasets.
    DataLoader turns our dataset into a Python iterable.
    More specifically, we want to turn our data into batches (or mini-batches).
    
    Why would we do this?
    1. It is more computationally efficient, as in, your computing hardware 
        may not be able to look (store in momery) at 60000 images in one hit.
        So we break it down to 32 images at a time (batch size of 32).
    2. It gives our neural network more chances to update its gradients per epoch.

'''
print(f"\n\ntrain_data:\n{train_data}\n\ntest_data:\n{test_data}")

from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn dataset into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

# Let's check out what we've created
print(f"\nDataLoaders: \n{train_dataloader}\n{test_dataloader}\n")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}\n")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"check first batch's shape: {train_features_batch.shape}, label shape: {train_labels_batch.shape}\n")

# # Show a sample in first batch
# torch.manual_seed(42)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.xlabel(f"Image size: {img.shape}, label: {label}, label size: {label.shape}")
# plt.show()

# Try plot a batch with {BATCH_SIZE} imgs
dataloader_iter = iter(train_dataloader)
random_idx = torch.randint(0, len(train_dataloader), size=[1]).item()

for _ in range(random_idx):
    next(dataloader_iter)

train_features_batch, train_labels_batch = next(dataloader_iter)

plt.figure(figsize=(10,6))
for i in range(32):
    plt.subplot(4,8,i+1)
    img = train_features_batch[i]
    label = train_labels_batch[i]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.suptitle(f"Batch {random_idx+1}")
plt.show()

debug=1