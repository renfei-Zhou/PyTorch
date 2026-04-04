### 6. The missing piece: non-linearity
'''
    "What patterns could you draw if you were given an infinite amount of a straight and non-straight lines?"
    Or in machine learning terms, an infinite (but really it is finite) of linear and non-linear functions?
'''
### 6.1 Recreating non-linear data (the red and blue circles)
# Make and plot data (for completeness)
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Convert data to tensors and then to train and test splits
import torch
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train[:5], y_train[:5]) # should show tensor



### 6.2 Build a model with non-linearity
'''
    Artifical neural networks are a large combination of linear and non-linear functions which are potentially able to find pattern in data.
'''
# Build a model with non-linearity activation functions
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()   # relu is a non-linear activation function
        
    def forward(self, x):
        # Where should we put our non-linear activation functions? 
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(f"\nmodel_3:\n{model_3}\n")

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),
                            lr=0.1)



### 6.3 Training a model with non-linearity
# Random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Download helper functions
import requests
from pathlib import Path
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skip downloading")
else:
    print("Downloading helper_functions.py")
    request = requests.get("http://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
from helper_functions import accuracy_fn

# Loop through data
epochs = 1000
for epoch in range(epochs):
    ### Training
    model_3.train()

    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))   # logits -> prediction probabilities -> prediction labels

    # 2. Calculate the loss
    loss = loss_fn(y_logits, y_train)   # BECWithLogitsLoss (takes in logits as first input)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        
    # Print out what's happenin'
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}  |   Loss: {loss:.4f}, Acc: {acc:.2f}%    |   Test_loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")






# 11_51_27 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 12_27_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
debug =1