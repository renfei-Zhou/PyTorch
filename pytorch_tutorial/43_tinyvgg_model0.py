'''
Video: PyTorch for Deep Learning & Machine Learning - Full Course
Section 04 Custom Datasets
    - 7. Data augmentation demo (TrivialAugmentWide)            ~22:30 - 22:50
    - 8. Model 0: TinyVGG WITHOUT data augmentation             ~22:50 - 23:30
    - 7.x train_step() / test_step() / train() functions        ~23:30 - 24:00
    - Plot loss curves, ideal loss curves / overfitting talk    ~24:00 - 24:15
      (aside: "Making Deep Learning Go Brrrr" blog on compute, ~23:00)
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, List, Tuple
from timeit import default_timer as timer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os


# Necessary data -------------------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
image_path = Path("data/pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"
# Necessary data ends here -------------------------------------



### 7. Other forms of transforms (data augmentation)
'''
    Data augmentation = artificially adding diversity to your training data.
    In the case of image data: applying various image transformations.
    This hopefully results in a model that's more generalizable to unseen data.

    torchvision.transforms.TrivialAugmentWide = augmentation strategy used to
    train recent state of the art torchvision models ("trivial" = random choice
    of augmentation at random strength, no search needed).
'''
train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),  # magnitude range 0-31
    transforms.ToTensor()
])
# (used for model_1 in 44 - shown here as a demo on random images in the video)



### 8. Model 0: TinyVGG without data augmentation
## 8.1 Creating transforms and loading data for Model 0
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# 1. Load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)
class_names = train_data_simple.classes

# 2. Turn the datasets into DataLoaders
# Setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = 0  # video: os.cpu_count() (Windows flat script -> 0)

# Create DataLoader's
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)


## 8.2 Create TinyVGG model class (same as 33 but for 3-channel 64x64 color images)
class TinyVGG(nn.Module):
    '''
    Model architecture copying TinyVGG from CNN explainer:
    https://poloclub.github.io/cnn-explainer/
    '''
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,  # 64x64 input -> two MaxPool2d halvings -> 16x16
                      out_features=output_shape)
        )

    def forward(self, x):
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))  # benefits from operator fusion

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,  # number of color channels in our image data
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

## 8.3 Try a forward pass on a single image (to test the model / find shape errors)
image_batch, label_batch = next(iter(train_dataloader_simple))
model_0(image_batch.to(device))

## 8.4 Use torchinfo to get an idea of the shapes going through our model
#   pip install torchinfo
try:
    from torchinfo import summary
    summary(model_0, input_size=[1, 3, 64, 64])
except ImportError:
    print("torchinfo not installed... pip install torchinfo")


## 8.5 Create train & test loop functions
'''
    train_step() - takes in a model and dataloader and trains the model on it
    test_step()  - takes in a model and dataloader and evaluates the model on it
    (this time they return (loss, acc) so we can track results per epoch)
'''
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device) -> Tuple[float, float]:
    # Put the model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)  # output model logits

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy metric (logits -> pred labels)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = device) -> Tuple[float, float]:
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


## 8.6 Create a train() function to combine train_step() and test_step()
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = device) -> Dict[str, List[float]]:
    # 1. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # 2. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # 3. Train step
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        # Test step
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # 4. Print out what's happening
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


## 8.7 Train and evaluate model 0
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_simple.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model_0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


## 8.8 Plot the loss curves of model 0
'''
    A loss curve is a way of tracking your model's progress over time.
    Ideal: loss goes down, accuracy goes up.
    - test loss much higher than train loss  -> OVERFITTING  (learning the training data too well)
    - both losses still high / going down    -> UNDERFITTING (model not learning enough yet)
    Ways to reduce overfitting: more data, data augmentation, regularization/dropout,
    simplify the model, transfer learning, early stopping.
    Ways to reduce underfitting: bigger model, train longer, lower regularization,
    transfer learning, tune learning rate.
'''
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

plot_loss_curves(model_0_results)
# plt.show()

# Save results for comparison in 44
import json
with open("model_0_custom_results.json", "w", encoding="utf-8") as f:
    json.dump(model_0_results, f)


debug=1
