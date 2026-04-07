### 8. Putting it all together with a multi-class classification problem
'''
    Binary classificaiton = 
        one thing or another (cat vs. dog, spam vs. not spam, ...)
    Multi-class classification =
        more than one thing or another (cat vs. dog vs. chicken)
'''
### 8.1 Creating a toy multi-class dataset
# Import dependencies
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,  # give the clusters a little shake-up
                            random_state=RANDOM_SEED)

# 2. Turn the data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# # 4. Plot data (visualize)
# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()



### 8.2 Building a multi-class classificaiton model in PyTorch
# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\ndevice: {device}\n")

# Build a multi-class classificaiton model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        ''' Initializes multi-class classification model.
        
        Args: 
            input_features (int): Number of input features to the model
            output_features (int): Number of output features (Number of output classes)
            hidden_units (int): Number of hidden units between layers, default 8
        '''
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=2,
                    output_features=4,  # equals to the number of the classes
                    hidden_units=8).to(device)
print(f"Check model_4:\n{model_4}\n")
print(f"Check if len(label)==len(classes): {'Yes' if NUM_CLASSES==len(torch.unique(y_blob_train)) else "No"}\nNumber of classes: {NUM_CLASSES}\ny_blob_train: {torch.unique(y_blob_train)}\n")



### 8.3 Create a loss function and an optimizer for a multi-class classification model
# Create a loss function for multi-class classification - loss function measures how wrong our model's predictions are
loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification - optimizer updates our model's parameters to try and reduce the loss
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1) # learning rate is a hyper-parameter



### 8.4 Getting prediction pobabilities for a multi-class PyTorch model
'''
    In order to evaluate and train and test our model, 
    we need to convert our model's outputs (logits) to prediction probabilities
    and then to prediction labels.

    Logits (raw output of the model) -> Pred probs (use torch.softmax) -> Pred labels (take the argmax of the prediction probabilities)
'''
# Let's get some raw outputs of our model (logits)
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))
    print(f"raw outputs:\n{y_logits[:5]}\nCompare to labels:\n{y_blob_test[:5]}\n")

    # 1. Convert our model's logit outputs to prediction probabilities
    y_pred_probs = torch.softmax(y_logits, dim=1)
    print(f"y_logits:\n{y_logits[:5]}\nConvert to pred.Probs:\n{y_pred_probs[:5]}\n")

    # 2. Convert our model's prediction probabilities to prediction labels
    y_preds = torch.argmax(y_pred_probs, dim=1)
    print(f"Now convert to pred.Labels:\n{y_preds[:10]}\n")



### 8.5 Creating a training loop and testing loop for a multi-class PyTorch model
# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
from helper_functions import accuracy_fn, plot_decision_boundary

# Set number of epochs
epochs = 100

# Put data to the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# Loop throght data
for epoch in range(epochs):
    ### Training
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)
        
    # Print out what's happenin'
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}  |   Loss: {loss:.4f}, Acc: {acc:.2f}%    |   Test_loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")



### 8.6 Making and evaluating predictions with a PyTorch multi-class model
# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# Go from logits -> Prediction probabilities -> pred labels
y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_pred_probs, dim=1)

# # Plot
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_4, X_blob_train, y_blob_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_4, X_blob_test, y_blob_test)
# plt.show()




### 9. A few more classification matrics... (to evaluate our classification model)
'''
    * Accuracy - out of 100 samples, how many does our model get right?
    * Precision
    * Recall
    * F1-score
    * Confusion matrix
    * Classification report
'''
from torchmetrics import Accuracy

# Setup metric
torchmetric_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

# Calculate accuracy
torchmetric_acc_result = torchmetric_acc(y_preds, y_blob_test)
print(f"\nAccuracy form torchmetrics:\n{torchmetric_acc_result}\n")


debug=1