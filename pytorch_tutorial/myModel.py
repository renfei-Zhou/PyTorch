import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# hyperparameter
lr = 0.01
epochs = 500

# Create *known* parameters
ideal_K0 = 22.0
ideal_K = 5.0
ideal_T = 200.0

# Create data
start = 0
end = 1000
step = 0.1

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = ideal_K * (1 - np.exp(-X / ideal_T)) + ideal_K0


# create training and testing data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


'''
Build model
'''

# Create linear regression model class
class myModel(nn.Module): # <- almost everything in pytorch inherhits from nn.Module
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_K = nn.Parameter(torch.tensor([4.0]) + 0.5 * torch.randn(1,
                                                                              requires_grad=True,
                                                                              dtype=torch.float))
        self.weights_T = nn.Parameter(torch.tensor([250.0]) + 50.0 * torch.randn(1,
                                                                              requires_grad=True,
                                                                              dtype=torch.float))
        self.bias_K0 = nn.Parameter(torch.tensor([18.0]) + 2.0 * torch.randn(1,
                                                                              requires_grad=True,
                                                                              dtype=torch.float))

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights_K * (1 - torch.exp(-x / self.weights_T)) + self.bias_K0



'''
Checking the contents of our PyTorch model
'''
# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Nodule)
model_0 = myModel()

# Check the parameters
model_0_params = list(model_0.parameters())

# Check parameters list
model_0_params_list = model_0.state_dict()




# Setup a loss function
loss_fn = nn.L1Loss()

# Setup a optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=lr) # lr = learning rate = hyperparameter you can set



# An epoch is one loop through the data... (this is a hyperparameter, we set it ourselves)
# epochs = 2000

# Track difference values
epoch_count = []
train_loss_values = []
test_loss_values = []


### Training
# 0. Loop through the data
for epoch in range(epochs):
    #  Set the model to training mode
    model_0.train() # train mode in PyTroch sets all parameters that require gradients to require gradients

    # 1. Forward pass
    y_preds = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_preds, y_train) # loss_fn = nn.L1Loss() defined before

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # by default how the optimizer changes will acculumate through the loop so... we have to zero them in step 3 for the next iteration of the loop 


    ### Testing
    model_0.eval() # turns off differnet settings in the model not needed for testing/evaluaiton
    with torch.inference_mode(): # turns off gradient tracking
        # 1. forward pass in testing
        test_preds = model_0(X_test)
        
        # 2. calculate the loss
        test_loss = loss_fn(test_preds, y_test)

    # print what's happing 
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())

        print(f"epochs: {epoch}  |  Loss:{loss}  |  Test loss:{test_loss}")
        print(model_0.state_dict())



predictions = test_preds.detach().numpy()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# --- Subplot 1: Loss curves ---
axs[0].plot(epoch_count, train_loss_values, label="Train loss")
axs[0].plot(epoch_count, test_loss_values, label="Test loss")
axs[0].set_title("Training and Testing Loss Curves")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend(prop={"size": 12})
axs[0].grid(True)

# --- Subplot 2: Predictions vs data ---
axs[1].scatter(X_train, y_train, c='b', s=4, label="Training Data")
axs[1].scatter(X_test, y_test, c='g', s=4, label="Testing Data")

axs[1].scatter(X_test, predictions, c='r', s=4, label="Predictions")

axs[1].set_title("Predictions vs Actual Data")
axs[1].legend(prop={"size": 12})
axs[1].grid(True)

plt.tight_layout()
plt.show()




debug=1
