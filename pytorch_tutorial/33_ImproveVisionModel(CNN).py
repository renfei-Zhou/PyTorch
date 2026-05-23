import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
from my_helper_functions import accuracy_fn, print_train_time, train_step, test_step, eval_model
# Necessary data -------------------------------------
torch.manual_seed(42)
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
# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
# Necessary data ends here -------------------------------------



### Model 2: Buildiing a Convolutional Neural Network (CNN)
'''
    CNN's are also known ConvNets.
    CNN's are known for their capability to find patterns in visual data.
    To find what's happening inside a CNN: https://poloclub.github.io/cnn-explainer/
'''
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    '''
    Model architecture that replicates the TinyVGG
    model from CNN explainer website.
    '''
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create a conv layer - http://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # 3x3 is the most common kernel size
                      stride=1, # how many pixels the kernel moves at a time
                      padding=1), # Values we can set ourselves in our NN's are called HYPERparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # there's a trick to calculate this...
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"shape for conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"shape for conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"shape for classifier: {x.shape}")
        return x
    

model_2 = FashionMNISTModelV2(input_shape=1, 
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)



### 7.1 Setting through nn.Conv2d()
# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(f"Test image:\n{test_image}")


# Create a single conv2d layer
conv_layer = nn.Conv2d(in_channels=3,       # input = 3 channels(RGB)
                       out_channels=10,     # output = 10 characteristic pictures
                       kernel_size=(3, 3),  # using 3*3 window to scan the picture
                       stride=1,            # moving 1 pixil per time
                       padding=1)           # 填充：给图像四周补一圈 0，让输出尺寸和输入一样

# Pass the data through the convolutional layer
conv_output = conv_layer(test_image.unsqueeze(0))
print(conv_output.shape)



### 7.2 Stepping through [nn.MaxPool2d]
# Print out original image shape without unsqueezed dimension
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(0).shape}")

# Create a sample nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")


# Getting more familiar with maxPool2d:
# Create a random tensor with a similar number of dimensions to our images
random_tensor = torch.randn(size=(1,1,2,2))
print(f"\nRandom tensor:\n{random_tensor}")
print(f"\nRandom tensor shape:\n{random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n {max_pool_tensor}")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")



### 7.3 Setup a loss function and optimizer for model_2
# Setup loss function/eval metrics/optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)


### 7.4 Training and testing model_2 using our training and test functions
# Measure time
train_time_start_model_2 = timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n--------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(train_time_start_model_2,
                                            train_time_end_model_2,
                                            device=device)

# Get model_2 results
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)
print(f"model_2 results: {model_2_results}")




debug=1
# 13_48_09 (PyTorch for Deep Learning & Machine Learning – Full Course)
# 14_51_41 (2026-04-14)
# 15_21_15 (2026-04-15)
# 15_35_55 (2026-04-20)
# 16_06_00 (2026-04-27)
# 16_25_08 (2026-04-29)
# 17_09_23 (2026-05-05)
# 17_37_23 (2026-05-13)
# 17_44_05 (2026-05-20)