import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

'''
    MaxPool(最大池)
    dilation(空洞卷积=插空值)
    Floor:   2.31 -> 2
    Ceiling: 2.31 -> 3
'''

# input setting
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1,1,5,5)) # -1 means let it calculate the size itself
print("\n input size: ", input.shape, "\n", pd.DataFrame(input.squeeze()))


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)    
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output
    

mymodel = MyModel()
output = mymodel(input)
print("\noutput:\n", pd.DataFrame(output.squeeze()))

'''
    the MaxPool take the max value in kernel size to reduce the data amount

'''


debug=1