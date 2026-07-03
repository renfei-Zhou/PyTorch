'''
Video: PyTorch for Deep Learning & Machine Learning - Full Course
Section 04 Custom Datasets
    - 11. Making a prediction on a custom image               ~24:55 - 25:30
        11.1 Load in a custom image with PyTorch
        11.2 Predict on the custom image with a trained model
        11.3 Put it all together: pred_and_plot_image()
    - Section wrap-up + exercises                             ~25:30 - 25:37 (end of video)

Key takeaway: data needs to be in the SAME format the model was trained on:
    - tensor, dtype float32
    - same shape (64x64x3)
    - same device
    - values scaled to [0, 1]
'''
import torch
from torch import nn
import torchvision
from torchvision import transforms
from pathlib import Path
from typing import List
import requests
import matplotlib.pyplot as plt


# Necessary data -------------------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
data_path = Path("data/")
class_names = ["pizza", "steak", "sushi"]

# TinyVGG (copied from 43) - needs trained weights: re-run training in 43/44 or load a state_dict
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*16*16, output_shape))
    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
# model_1.load_state_dict(torch.load("models/04_pytorch_custom_datasets_model_1.pth"))
# Necessary data ends here -------------------------------------



### 11.1 Loading in a custom image with PyTorch
# Download custom image (a photo of pizza)
custom_image_path = data_path / "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download...")

# Read in custom image with torchvision.io (imports uint8, values 0-255)
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
print(f"Custom image tensor:\n {custom_image_uint8}")
print(f"Custom image shape: {custom_image_uint8.shape}")   # [3, 4032, 3024]
print(f"Custom image dtype: {custom_image_uint8.dtype}")   # torch.uint8

# Plot it (permute CHW -> HWC for matplotlib)
plt.imshow(custom_image_uint8.permute(1, 2, 0))
# plt.show()



### 11.2 Making a prediction on a custom image with a trained PyTorch model
# Trying to predict raw uint8 image -> ERROR: model expects float32 in [0, 1]
# Load in the custom image and convert to float32 + scale to [0, 1]
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 255.

# Model was trained on 64x64 images -> ERROR if shape doesn't match: transform/resize first
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
])
custom_image_transformed = custom_image_transform(custom_image)
print(f"Original shape: {custom_image.shape}")
print(f"Transformed shape: {custom_image_transformed.shape}")

# Predict: also needs a BATCH dimension (unsqueeze) and to be on the right DEVICE
model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
print(custom_image_pred)  # logits

# logits -> prediction probabilities -> prediction labels
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction: {class_names[custom_image_pred_label.cpu()]}")



### 11.3 Putting custom image prediction together: building a function
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image with a trained model and plots the image and prediction."""
    # Load in the image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    # Make sure the model is on the target device
    model.to(device)

    # Turn on eval/inference mode and make a prediction
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image (batch dimension)
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on the image with an extra dimension
        target_image_pred = model(target_image.to(device))

    # Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0))  # remove batch dim and rearrange to HWC
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

# Pred on our custom image
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
# plt.show()


'''
Section 04 (and the video) ends here (~25:37).
Main takeaways:
    1. PyTorch has in-built functions for common datasets, but you can build
       a Dataset for almost anything by subclassing torch.utils.data.Dataset
    2. Data augmentation (e.g. TrivialAugmentWide) can help reduce overfitting
    3. To predict on custom data: SAME dtype / shape / device as training data
Exercises + extra-curriculum: https://www.learnpytorch.io/04_pytorch_custom_datasets/
Next chapters (not in this video): 05 Going Modular, 06 Transfer Learning...
'''

debug=1
