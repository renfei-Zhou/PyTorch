import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

'''
    Convolution(卷积) in torch.nn

    nn.conv2d
'''

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))

print("input size: ", input.shape)
print("kernel size: ", kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print("\noutput(stride=1):\n", pd.DataFrame(output1.squeeze()))

output2 = F.conv2d(input, kernel, stride=2)
print("\noutput(stride=2):\n", pd.DataFrame(output2.squeeze()))


'''
    padding: expand the matrix, 
    e.g 5x5 to 7x7, fill the blank as 0
    [[1,2,3],     [[0,0,0,0,0],
     [4,5,6],  to  [0,1,2,3,0],
     [7,8,9]]      [0,4,5,6,0],
                   [0,7,8,9,0],
                   [0,0,0,0,0]]
'''

output3 = F.conv2d(input,kernel, stride=1, padding=1)
print("\noutput(stride=1, padding=1):\n", pd.DataFrame(output3.squeeze()))


output4 = F.conv2d(input,kernel, stride=2, padding=1)
print("\noutput(stride=2, padding=1):\n", pd.DataFrame(output4.squeeze()))


output5 = F.conv2d(input,kernel, stride=3, padding=1)
print("\noutput(stride=3, padding=1):\n", pd.DataFrame(output5.squeeze()))


debug=1