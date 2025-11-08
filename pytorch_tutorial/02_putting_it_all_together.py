import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# check pytorch version
torch.__version__


# setup device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Data for linear regression
ideal_weight = 0.7
ideal_bais = 0.3

# create range value
start = 0
end = 1
step = 0.02

# create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # errors without unsqueeze
y = ideal_weight * X + ideal_bais


# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len_1,len_2,len_3,len_4 = len(X_train), len(X_test), len(y_train), len(y_test)

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


# Create a linear model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use nn.Linear() for creating the linear params y = x * A^T + b
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1) # 1 input, 1 output
    
    def forward(self, x:torch.Tensor) -> torch.Tensor: # x should be a torch Tensor, and the return should be a torch Tensor
        return self.linear_layer(x)
    

# Set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1_dict = model_1.state_dict()


# check the current using device
device = next(model_1.parameters()).device

# set the model to use target device
model_1.to(device)
target_device = next(model_1.parameters()).device




# Train the code (Loss_fun, Optimizer, Training_loop, Testing_loop)
loss_fn = nn.L1Loss() # same as MAE

optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.01)

torch.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. optmizer zero grad
    optimizer.zero_grad()

    # 4. perform backpropagation 
    loss.backward()

    # 5. optimizer step
    optimizer.step()

    ### Testing:
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)    
    
    # pring out what's happening
    if epoch % 10 == 0:
        print(f"epoch: {epoch}  |   Loss:{loss} |   Test_loss:{test_loss}")

# check how close the prediction
print(model_1.state_dict(),f"\nweight={ideal_weight}    |   bias={ideal_bais}")    
plot_predictions(predictions=test_pred)


### saving and loading the model
# saving
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "01_Linear_Regression_ModelV2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)


# loading
loaded_model_1 = LinearRegressionModelV2()

loaded_model_1_state = loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Make some predictions with our loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)

# Compare loaded model preds with original model preds
with torch.inference_mode():
    origin_model_preds = model_1(X_test)

print(origin_model_preds == loaded_model_preds) # True


debug=1