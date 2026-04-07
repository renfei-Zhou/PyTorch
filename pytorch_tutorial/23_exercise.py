### Exercise
'''
    Exercises:https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/exercises
    Solutions:https://github.com/mrdbourke/pytorch-deep-learning/tree/main/extras/solutions
'''

# 1. Create a random tensor with shape (7, 7)
import torch
X = torch.rand(size=(7,7))
print(f"Ex1:\nshape: {X.shape}\n{X}\n")

# 2. Perform a matrix multiplication on the tensor from Ex1 
#    with another random tensor with shape (1, 7)
Y = torch.rand(size=(1,7))
Z = torch.matmul(X,Y.T)
print(f"Ex2:\nshape: {Z.shape}\n{Z}\n")

# 3. Set the random seed to 0 and do Ex1 & 2 over again
torch.manual_seed(0)
X = torch.rand(size=(7,7))
Y = torch.rand(size=(1,7))
Z = torch.matmul(X, Y.T)
print(f"Ex3:\n{Z}\n")

# 4. Find the maximum and minimum (index) values of the output of Ex3
max = torch.max(Z)
min = torch.min(Z)
max_idx = torch.argmax(Z)
min_idx = torch.argmin(Z)
print(f"Ex4:\nmax: {max}\nmax idx: {max_idx}\nmin: {min}\nmin idx: {min_idx}\n")

# 5. Make a random tensor with shape (1, 1, 1, 10) 
# and then create a new tensor with 
# all the 1 dimensions removed to be left with a tensor of shape (10). 
# Set the seed to 7 when you create it and print out the first tensor 
# and it's shape as well as the second tensor and it's shape.
torch.manual_seed(7)
tensor_ex5 = torch.rand(size=(1,1,1,10))
tensor_ex5_new = tensor_ex5.squeeze()
print(f"Ex5:\nshape:{tensor_ex5.shape}\n{tensor_ex5}\nremoved:\nshape:{tensor_ex5_new.shape}\n{tensor_ex5_new}\n")






debug=1