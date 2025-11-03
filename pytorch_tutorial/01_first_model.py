'''
PyTorch workflow:
    1. data (prepare and load)
    2. build model
    3. fitting the model to data (training)
    4. making predictions and evalution a model (inference)
    5. saving and loading a model
    6. putting it all together
'''
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
'''
linear regression with *known* parameter
    y = a * x + b
    param a = weight
    param b = bias
'''

# Create *known* parameters
ideal_weight = 0.7
ideal_bias = 0.3


# Create data
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = ideal_weight * X + ideal_bias

'''
Split the data:
    Training set    ~60-80%     Always
    Validation set  ~10-20%     Often but not always
    Testing set     ~10-20%     Always
'''

# create training and testing data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# visualize the training data
def plot_predictions(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):
    '''
        Plots training data, testing data and compares predictions.
    '''
    plt.figure(figsize=(10,7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training Data")
    # Plot testing data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing Data")
    # Are there predictions
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")
    # legends, grid and show
    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()




'''
Build model
'''

# Create linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherhits from nn.Module
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights * x + self.bias


'''
PyTorch model building esentials
    torch.nn
        contains all of the buildings for computational graphs
        (a neural network canbe considered as a computational graph)
    torch.nn.Parameter
        what parameters should try and learn our model, 
        often a Pytroch layer from torch.nn will set these for us
    torch.nn.Module
        The base class for all neural network modules,
        if you subclass it, you should overwrite forward()
    torch.optim
        this where the optimizers in PyTorch live, they will help with gradient descent
    def forward()
        All nn.Module subclasses require you to overwrite forward(),
        this method defines what happens in the forward computation
'''




'''
Checking the contents of our PyTorch model
'''
# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Nodule)
model_0 = LinearRegressionModel()

# Check the parameters
model_0_params = list(model_0.parameters())

# Check parameters list
model_0_params_list = model_0.state_dict()




'''
Making prediction using `torch.inference_mode()`
    To check our model's predictive power, let's see how well it predicts `y_test` based on `X_test`.
    When we pass data through our model, it's going to run it through the `forward()` method.
'''

# # Make predictions with model
# with torch.inference_mode():
#     y_preds = model_0(X_test)

# # visualize the prediction
# plot_predictions(predictions=y_preds)


'''
Train model
    The whole idea of training is for a model to move 
        from some *unknown* parameters (these may be random) to some *known* parameters.
    Or in other words 
        from a poor representation of the data to a better representation of the data.

    One way to measure how poor or how wrong your models predictions are 
        is to use a loss function.
    
    **Note: Loss function may also be called cost function or criterion in different areas.**

    
    Things we need to train:
        Loss function: 
            To measure how wrong are the predictions to ideal, lower is better
        Optimizer:
            Takes into account the loss of a model and adjust the model's parameters

    And Specifically in PyTorch, we need:
        A training loop
        A testing loop
'''

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup a optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # lr = learning rate = hyperparameter you can set


'''
Build a training and testing loop
    things we need in a training loop:
        0. Loop through the data
        1. Forward pass (also called forward propagation) to make predictions on data
            this involves data moving through our model's `forward()` function
        2. Calculate the loss (compare forward pass predictions to ground truth labels)
        3. Optimizer zero grad
        4. Loss backward (**backpropagation**) - move backwards through the network to 
            calculate the gradients of the each of the parameters of our model with respect to the loss.
        5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss
'''

# An epoch is one loop through the data... (this is a hyperparameter, we set it ourselves)
epochs = 200

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


# Plot the loss curve
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and Testing Loss Curves")
plt.xlabel("Loss")
plt.ylabel("Epochs")

plt.legend(prop={"size": 14})
plt.grid(True)
# plt.show()





'''
Saving a model in PyTorch
    1. `torch.save()` -- allows saving a PyTorch object in Python's pickle format
    2. `torch.load()` -- allows loading a saved PyTroch object
    3. `torch.nn.Module.load_state_dict()` -- allows loading a model's saved state dictionary
'''
## saving this PyTorch model
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_Linear_Regression_Model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)



## Loading a PyTroch model
# To load a saved state_dict, we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# Load the saved state_dict of model_0 (update the new instance with new params)
loaded_model_0_state = loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# Compare loaded model preds with original model preds
with torch.inference_mode():
    origin_model_preds = model_0(X_test)

print(origin_model_preds == loaded_model_preds) # True




debug=1