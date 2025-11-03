import torch
from torch import nn
'''
    First look of the neural network using torch.nn
'''

class MyModel(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, input):
        output = input + 1
        return output
    

MyModel = MyModel()
x = torch.tensor(1.0)
output = MyModel(x)

debug=1
