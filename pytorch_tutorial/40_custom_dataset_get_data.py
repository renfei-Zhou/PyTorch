'''
Video: PyTorch for Deep Learning & Machine Learning - Full Course
Section 04 Custom Datasets - starts ~19:44
    - PyTorch domain libraries (torchvision/torchtext/torchaudio/torchrec)
    - 1. Get data (pizza, steak, sushi subset of Food101)   ~20:00 - 20:20
    - 2. Become one with the data (data preparation)        ~20:20 - 20:45
    - 3. Visualize an image with PIL / numpy                ~20:45 - 21:00

Goal of section 04: build a food vision model on our OWN dataset
(pizza/steak/sushi = 10% subset of the Food101 dataset).
'''
import torch
from pathlib import Path
import requests
import zipfile
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"



### 1. Get data (pizza, steak and sushi images from Food101)
# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak and sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak and sushi data...")
        zip_ref.extractall(image_path)



### 2. Becoming one with the data (data preparation and data exploration)
def walk_through_dir(dir_path):
    '''Walks through dir_path returning its contents.'''
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)
# data/pizza_steak_sushi/
#   train/ (pizza 78, steak 75, sushi 72)
#   test/  (pizza 25, steak 19, sushi 31)

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"



### 3. Visualizing an image
'''
    1. Get all of the image paths
    2. Pick a random image path using random.choice()
    3. Get the image class name using pathlib.Path.parent.stem
    4. Open the image with Python's PIL
    5. Show the image and print metadata
'''
random.seed(42)

# 1. Get all image paths (glob together all .jpg files in nested dirs)
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Pick a random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory
#    where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
# img.show()

# Turn the image into a numpy array and visualize with matplotlib
img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels] (HWC)")
plt.axis(False)
# plt.show()


debug=1
