'''
Video: PyTorch for Deep Learning & Machine Learning - Full Course
Section 04 Custom Datasets
    - 4. Transforming data with torchvision.transforms      ~21:00 - 21:20
    - 5. Loading image data using ImageFolder (option 1)    ~21:20 - 21:40
    - Turning loaded images into DataLoader's

Option 1: load image data with torchvision.datasets.ImageFolder
(works when data is in "standard image classification format": class-name folders)
'''
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image


# Necessary data -------------------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
image_path = Path("data/pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"
image_path_list = list(image_path.glob("*/*/*.jpg"))
# Necessary data ends here -------------------------------------



### 4. Transforming data with torchvision.transforms
'''
    Before we can use our image data with PyTorch:
    1. Turn the target data into tensors (numerical representation of images)
    2. Turn it into torch.utils.data.Dataset (and later DataLoader)
'''
# Write a transform for turning images into tensors
data_transform = transforms.Compose([
    # Resize our images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal (a form of data augmentation)
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch.Tensor (also scales pixel values 0-255 -> 0.0-1.0)
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    '''
    Selects random images from a path of images and loads/transforms
    them, then plots the original vs the transformed version.
    '''
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            # Transform and plot target image
            transformed_image = transform(f).permute(1, 2, 0)  # note: matplotlib needs HWC, transform gives CHW
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        n=3,
                        seed=42)
# plt.show()



### 5. Option 1: Loading image data using torchvision.datasets.ImageFolder
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,       # a transform for the data
                                  target_transform=None)          # a transform for the label/target
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
print(train_data, test_data)

# Get class names as list
class_names = train_data.classes
print(class_names)  # ['pizza', 'steak', 'sushi']

# Get class names as dict
class_dict = train_data.class_to_idx
print(class_dict)   # {'pizza': 0, 'steak': 1, 'sushi': 2}

# Check the lengths of our dataset
print(len(train_data), len(test_data))  # 225, 75

# Index on the train_data Dataset to get a single image and label
img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n {img}")
print(f"Image shape: {img.shape}")          # torch.Size([3, 64, 64]) -> CHW
print(f"Image datatype: {img.dtype}")       # torch.float32
print(f"Image label: {label}")              # 0
print(f"Label datatype: {type(label)}")     # int

# Rearrange the order of dimensions to plot with matplotlib (CHW -> HWC)
img_permute = img.permute(1, 2, 0)
plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis("off")
plt.title(class_names[label], fontsize=14)
# plt.show()


# Turn loaded images into DataLoader's
'''
    A DataLoader helps turn our Dataset's into iterables (batches),
    so the model can see one batch at a time (fits in memory + more gradient
    descent steps per epoch).
'''
import os
BATCH_SIZE = 1
# video: NUM_WORKERS = os.cpu_count()  (Windows flat script -> use 0 to avoid multiprocessing issues)
NUM_WORKERS = 0

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)
print(len(train_dataloader), len(test_dataloader))

img, label = next(iter(train_dataloader))
# Batch size will now be 1, you can change the batch size if you like
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")


debug=1
