import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

'''
Neural Network classification with PyTorch

    Classification is a problem of predicting whether something is one thing or another 
    (there can be multiple things as the options).

    Notebook:https://www.learnpytorch.io/02_pytorch_classification/
'''



'''
1. Make classification data and get it ready
'''
# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

length_X = len(X)
length_y = len(y)
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")
debug=1

# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head(10))
debug=1

# # Virsualize
# plt.scatter(x=X[:,0],
#             y=X[:,1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()
# debug=1


'''
These two circles are our toy dataset, small enough + still sizeable 

    More toy datasets: scikit-learn toy datasets
'''



### 1.1 Check input and output shapes
shape_X = X.shape
shape_y = y.shape

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
debug=1


### 1.2 Turn data into tensors and create train and test splits
torch.manual_seed(42)
# trun X form type(X)=numpy.ndarray to tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(f"First 5 samples of tensor X:\n {X[:5]}")
print(f"First 5 samples of tensor y:\n {y[:5]}")
debug=1

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% of data will be test, 80% be train
                                                    random_state=42)

length_X_train = len(X_train)
length_X_test = len(X_test)
length_y_train = len(y_train)
length_y_test = len(y_test)
debug=1



'''
2. Building a model
    Build a model to classify the blue and red dots
    To do so, we want to:
        1. Setup device agonistic code so our code will run on an accelerator (GPU)
        2. Construct a model (by subclassing 'nn.Module')
        3. Define a loss function and optimizer 
        4. Create a training and testing loop
'''
'''
    Create a model:
    1. Subclasses 'nn.Module' (almost all models in PyTorch subclass 'nn.Module')
    2. Create 2 'nn.Module()' layers that are capable of handling the shapes of our data
    3. Defines a 'forward()' method that outlines the forward pass (or forward computation) of the model
    4. Instatiate an instance of our model class and send it to the target 'device'
'''

# 1 Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 2  Create 2 'nn.Module()' layers 
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

    # 3 Define a forward() method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # data x --> layer_1 --> layer_2 --> output
            
# 4 Instatiate an instance of our model class and send it to the target 'device'
device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = CircleModelV0().to(device)



'''
    Let's replicate the model above using nn.Sequential()
'''
# # method 1
# model_1 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)

# # method 2
# class CircleModelV0(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.two_linear_layers = nn.Sequential(
#             nn.Linear(in_features=2, out_features=5),
#             nn.Linear(in_features=5, out_features=1)
#         )
#     def forward(self, x):
#         return self.two_linear_layers(x)
# model_1 = CircleModelV0().to(device)


# Make predictions
model_0_state_dict = model_0.state_dict()
print(model_0_state_dict)

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions: \n{torch.round(untrained_preds[:10])}")
print(f"\nFirst 10 labels: \n{y_test[:10]}")



'''
2.1 Setup loss function and optimizer

Which loss function and optimizer should you use?
    For regression you might want MAE or MSE (meas absolute error and mean squared error).
    For classification you might want  (cross entropy).
    * For loss-fn wr're using `torch.nn.BCEWithLogitsLoss()`
'''
# Setup the loss-fn
# loss_fn = nn.BCELoss() # requires inputs to have gone through the sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function built-in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



'''
3. Train model
    need to build a training loop:
        1. Forward pass
        2. Calculate the loss
        3. Optimizer zero grad
        4. Loss backward (backpropagation)
        5. Optimizer (gradient descent)

### 3.1 Going from raw logits --> prediction probabilities --> prediction labels

    Our model outputs are going to be raw **logits**.

    We can convert these **logits** into **prediction probablities** by passing them to 
    some kind of activation functions (e.g. sigmoid for binary cross entropy and softmax for
    multi-class classification).

    Then we can convert our model's prediction probablities to **prediction labels** 
    by either rounding them or taking the `argmax()`.
'''
# View the first 5 outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
    print(f"y_logits:\n{y_logits}")

# Use the sigmoid activation function on our model logits to turn them into prediction probablities
y_pred_probs = torch.sigmoid(y_logits)
print(f"y_pred_probs: \n{y_pred_probs}\ny_pred_probs_round: \n{torch.round(y_pred_probs)}")

'''
For our prediction probability values, we need to perform a range-style rounding on them:
    y_pred_probs >= 0.5, y=1 (class 1)
    y_pred_probs < 0.5, y=0 (class 0)
'''
# Find the prediction labels
y_preds = torch.round(y_pred_probs)

# In full (logits --> pred pros --> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(f"Check equality:\n{torch.eq(y_preds.squeeze(), y_pred_labels.squeeze())}")




### 3.2 Building a training and testing loop
# Set epochs
epochs = 100

# Put data to target the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Building training and evaluation loop
for epoch in range(epochs):
    ### Training 
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze() # squeeze() removes an extra dimension from a Tensor
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits --> pred probility --> pred labels

    # 2. Calculation loss/accuracy
    loss = loss_fn(y_logits, # nn.BCEWithLogitsLoss expects raw logits as input
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (Backpropagation)
    loss.backward()

    # 5. Optimizer step (Gradient descent)
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/acc
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}  |   Loss: {loss:.5f}, Acc: {acc:.2f}%   |   Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")



'''
### 4. Make predictions and evalate the model

    From the matrics it looks like our model isn't learning anything...

    So to inspect it let's make some predictions and make them visual.
'''
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skip downloading")
else:
    print("Downloading helper_functions.py")
    request = requests.get("http://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
import matplotlib.pyplot as plt

# plt.figure(figsize=(12,6))

# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0, X_train, y_train)

# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0, X_test,y_test)

# plt.show()




'''
### 5. Improve a model (from a model perspective)

    * Add more layers - give the model more chances to learn about patterns in the data
    * Add more hidden layers - go from 5 to 10 hidden units
    * Fit for longer
    * Changing the activation functions
    * Change the learning rate

    These options are all from a model's perspective because they deal directly with the model,
    rather than the data.

    And because of these options are all values we can change, they are referred as **hyperparameters**.

    Let's try and improve our model by:
        * Adding more hidden units: 5 --> 10
        * Increase the number of layers: 2 --> 3
        * Increase the number of epochs: 100 --> 1000
    ! Change one value at a time and track the results (experiment tracking and machine learning)
'''
# Check adding layers
class CircleModelV1(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z              # simple way
        return self.layer_3(self.layer_2(self.layer_1(x))) # write in one line --> faster

model_1 = CircleModelV1().to(device)
model_1_dict = model_1.state_dict()

# Create a loss fn
loss_fn = nn.BCEWithLogitsLoss()

# Create an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

# Write a training and evaluation loop for model_1
epochs = 1000

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train()
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))   # logits -> pred probability -> prediction labels

    # 2. Calculate the loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards (backpropagation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculate loss
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    
    # Print epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}  |   Loss: {loss:.5f}, Acc: {acc:.2f}%    |   Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")



debug =1