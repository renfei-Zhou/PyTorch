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



#### 3.1 Setup loss, optimizer and evaluation matrics   
'''
    Loss function:
        since we're working with multi-class data, our loss function will be nn.CrossEntropyLoss()
    Optimizer:
        our optimizer torch.optim.SGD() (stochastic gradient descent)
    Evaluaiton matric:
        since we're working on a classification problem, let's use accuracy as our evaluation metric
'''
import requests
from pathlib import Path

# Download help function form Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download...")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# Import accuracy metric
from helper_functions import accuracy_fn

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)



### 3.2 Creating a function to time out our experiments
'''
    Machine learning is very experimental.

    Two of the main things you'll often want to track are:
        1. Model's performance (loss and accuracy values etc)
        2. How fast it runs
'''
from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    ''' Print difference between start and end time. '''
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# # test
# start_time = timer()
# import time
# time.sleep(1)
# end_time = timer()
# print_train_time(start_time, end_time, "cpu")



### 3.3 Creating a training loop and training a model on batches of data
'''
    1. Loop through epochs.
    2. Loop through training batches, perform training steps, calculate the train loss per batch.
    3. Loop through testing batches, perform testing steps, calculate the test loss per batch.
    4. Print out what's happening.
    5. Time it all (for fun).
'''
# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set number of epochs (we'll keep it small for faster training time)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    ### Training
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate the loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out what's happening
        if batch % 400 == 0:
            print(f"Looed at {batch * len(X)}/{len(train_dataloader.dataset)}")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader) 

    ### Testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X_test)

            # 2. Calculate the loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # 3. Calculate accuracy
            test_acc += accuracy_fn(y_true=y_test,
                                    y_pred=test_pred.argmax(dim=1))
            
        # Calcualte the test loss average per batch
        test_loss /= len(test_dataloader)

        # Cauclate the test acc average per batch
        test_acc /= len(test_dataloader)

    # Pring out what's happening
    print(f"\nTrain loss: {train_loss:.4f}  |   Test loss: {test_loss:.4f}   |   Test acc: {test_acc:.4f}")

# Calculate train time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(train_time_start_on_cpu, 
                                            train_time_end_on_cpu, 
                                            str(next(model_0.parameters()).device))


### 4. Make predictions and get Model_0 results
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    '''
        Returns a dictionary containing the results of model predicting on data_loader.
    '''
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # Only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn) 
print(model_0_results)


debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_46_03
# 14_51_41 (2026-04-14)
# 15_21_15 (2026-04-15)
# 15_35_55 (2026-04-20)
# 16_06_00 (2026-04-27)
# 16_25_08 (2026-04-29)
