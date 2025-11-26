import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_circles


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

# length_X = len(X)
# length_y = len(y)
# print(f"First 5 samples of X:\n {X[:5]}")
# print(f"First 5 samples of y:\n {y[:5]}")
# debug=1

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



