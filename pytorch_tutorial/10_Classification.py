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


### 1. Make classification data and get it ready

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

# # Make DataFrame of circle data
# circles = pd.DataFrame({"X1": X[:, 0],
#                         "X2": X[:, 1],
#                         "label": y})
# print(circles.head(10))
# debug=1

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


### 2. Building a model
'''
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

# 2.1 Construct a model that subclasses nn.Module
class CircleModel1V0(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 2.2  Create 2 'nn.Module()' layers 
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

        # 2.3 Define a forward() method that outlines the forward pass
        def forward(self, x):
            return self.layer_2(self.layer_1(x)) # data x --> layer_1 --> layer_2 --> output
            





