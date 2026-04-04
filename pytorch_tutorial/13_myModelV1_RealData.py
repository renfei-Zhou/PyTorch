import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_data_sim(trace):
    # Read CSV
    df_events = pd.read_csv(f"data/local_test/{trace}_events.csv")
    df_bounds = pd.read_csv(f"data/local_test/{trace}_bounds.csv")

    # Convert string column to timezone-aware datetime
    df_events["events_time"] = pd.to_datetime(df_events["events_time"], utc=True).dt.tz_convert("Europe/Berlin")
    df_bounds["start_bounds"] = pd.to_datetime(df_bounds["start_bounds"], utc=True).dt.tz_convert("Europe/Berlin")
    df_bounds["end_bounds"] = pd.to_datetime(df_bounds["end_bounds"], utc=True).dt.tz_convert("Europe/Berlin")

    # Get data
    events_time = df_events["events_time"]
    temperature_T1 = df_events["temperature_T1"].values

    start_bounds = df_bounds["start_bounds"]
    end_bounds = df_bounds["end_bounds"]

    return events_time, temperature_T1, start_bounds, end_bounds

# hyperparameter
trace = "Trace526166"
lr = 0.01
epochs = 500


def create_equidistant_data(X, y):
    # Create equidistant training indices (80% of total)
    num_train = int(0.8 * len(X))

    # Select evenly spaced indices of training
    train_indices = torch.linspace(0, len(X)-1, num_train).long()

    # Ctrate boolean mask for train/test split
    mask = torch.zeros(len(X), dtype=torch.bool)
    mask[train_indices] = True

    # Apply mask to get train/test data
    X_train, y_train = X[mask], y[mask]
    X_test, y_test = X[~mask], y[~mask]

    return X_train, y_train, X_test, y_test



# Create linear regression model class
class myModel(nn.Module): # <- almost everything in pytorch inherhits from nn.Module
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_K = nn.Parameter(torch.randn(1) * 8.0) # K in [0, 8)
        self.weights_T = nn.Parameter(400.0 + torch.randn(1) * 100.0) # T in [400, 500)
        self.bias_K0 = nn.Parameter(19.0 + torch.randn(1) * 4.0) # K0 in [19, 23)

        self.softplus = nn.Softplus()  # ensures positive K and T

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights_K * (1 - torch.exp(-x / self.weights_T)) + self.bias_K0



# Create a random seed
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Nodule)
model_0 = myModel()

# Setup a loss function
loss_fn = nn.MSELoss()

# Setup a optimizer (stochastic gradient descent)
optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=lr) # lr = learning rate = hyperparameter you can set





# Prepare the data
events_time, temperature_T1, start_bounds, end_bounds = get_data_sim(trace)


for idx, (start, end) in enumerate(zip(start_bounds, end_bounds)):
    # Track difference values
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    mask = (events_time >= start) & (events_time <= end)
    signal_value = temperature_T1[mask]
    signal_time = (events_time[mask] - events_time[mask].iloc[0]).dt.total_seconds().to_numpy()
    
    # interpolation
    num_interp_points = 1000  # adjust as needed (e.g. 500–1000)
    time_uniform = np.linspace(signal_time[0], signal_time[-1], num_interp_points)

    # Create interpolation function
    f_interp = interp1d(signal_time, signal_value, kind='linear', fill_value='extrapolate')
    value_uniform = f_interp(time_uniform)

    # === Convert to torch tensors ===
    signal_time_t = torch.tensor(time_uniform, dtype=torch.float32)
    signal_value_t = torch.tensor(value_uniform, dtype=torch.float32)

    X_train, y_train, X_test, y_test = create_equidistant_data(signal_time_t, signal_value_t)
    
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)



    ### Start Training
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
        if epoch % 50 == 0:
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
