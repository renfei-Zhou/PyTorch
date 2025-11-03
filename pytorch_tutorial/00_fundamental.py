import torch
import numpy as np
import pandas as pd


def pytorch_version():
    print(torch.__version__)


def scalar_vector_matrix_TENSOR():
    scalar = torch.tensor(7)
    print(f"\nscalar: \n{scalar}")
    print(f"scalar dimension: {scalar.ndim}")
    print(f"got tensor back in python: {scalar.item()}")

    vector = torch.tensor([1,2,3])
    print(f"\nvector: \n{vector}")
    print(f"vector dimension: {vector.ndim}")
    print(f"vector shape: {vector.shape}")

    matrix = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(f"\nmatrix: \n{matrix}")
    print(f"matrix dimension: {matrix.ndim}")
    print(f"matrix shape: {matrix.shape}")
    print(f"matrix row1: {matrix[0]}")
    print(f"matrix row2: {matrix[1]}")
    print(f"matrix row3: {matrix[2]}")

    TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
    print(f"\nTENSOR: \n{TENSOR}")
    print(f"TENSOR dimension: {TENSOR.ndim}")
    print(f"TENSOR shape: {TENSOR.shape}")
    print(f"\nTENSOR[0][0][0]= {TENSOR[0][0][0]}")
    print(f"TENSOR[0][0][1]= {TENSOR[0][0][1]}")
    print(f"TENSOR[0][1][0]= {TENSOR[0][1][0]}")
    print(f"TENSOR[0][2][2]= {TENSOR[0][2][2]}")


def random_tensor():
    '''
        Why random tensors?

        Random tensors are important because 
        the way many neural networks learn is that 
        they start with tensors full of random numbers and 
        then adjust those random numbers to better represent the data.

        Start with random numbers -> 
        look at data -> 
        update random nubmers -> 
        look at data -> 
        update random numbers
    '''

    # Create a random tensor of size (3, 4)
    random_tensor = torch.rand(3, 4)
    print(f"\nrandom_tensor: \n{random_tensor}")

    # Create a random tensor with similar shape to an image tensor
    random_image_size_tensor = torch.rand(224, 224, 3) # height, width, color channel (R, G, B)
    print(f"\nrandom_image_size_tensor shape: \n{random_image_size_tensor.shape}")
    print(f"\nrandom_image_size_tensor dimension: \n{random_image_size_tensor.ndim}")


def zeros_and_ones():
    # Create a tensor of all zeros
    zeros = torch.zeros(3, 4)
    print(f"\nzeros: \n{zeros}")

    # Create a tensor of all ones
    ones = torch.ones(3, 4)
    print(f"\nones: \n{ones}")


def range_of_tensors_and_tensors_like():
    # Use torch.range() and get deprecated message, use torch.arange()
    one_to_ten = torch.arange(start=1, end=11, step=1) # [a,b),step
    print(f"\none_to_ten: \n{one_to_ten}")

    # Creating tensors like
    one_to_ten_like = torch.zeros_like(input=one_to_ten)
    print(f"\nzeros like: \n{one_to_ten_like}")
    one_to_ten_like = torch.ones_like(input=one_to_ten)
    print(f"\nones like: \n{one_to_ten_like}")


def tensor_datatypes():
    """
        Tensor datatypes is one of the 3 big errors 
        you'll run into with PyTorch & deep learning:

            1. Tensors not right datatype
            2. Tensors not right shape
            3. Tensors not on the right device
    """

    # Float 32 tensor
    float_32_tensor = torch.tensor([3, 6.0, 9.0], dtype=None, # float32 or float16
                                                  device=None, # "cup" or "gpu"
                                                  requires_grad=True) # whether or not to track gradients with this tensors operations
    print(f"\nfloat_32_tensor: \n{float_32_tensor}")
    print(f"\ndtype: \n{float_32_tensor.dtype}")

    # change to float16
    float_16_tensor = float_32_tensor.type(torch.float16)
    print(f"\nchange to float16: \n{float_16_tensor}")

    # multiplication:  float_16_tensor * float_32_tensor
    print(f"\nmultiplication:  float_16_tensor * float_32_tensor \n{float_16_tensor * float_32_tensor}")

    # Int 32 tenser
    int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
    print(f"\nint_32_tensor: \n{int_32_tensor}")

    # multiplication:  float_32_tensor * int_32_tensor
    print(f"\nmultiplication:  float_32_tensor * int_32_tensor \n{float_32_tensor * int_32_tensor}")


def getting_information_from_tensors():
    '''
        1. Tensors not right datatype 
            to get datatype from a tensor, can use `tensor.dtype`

        2. Tensors not right shape 
            to get shape from a tensor, can use `tensor.shape`
        
        3. Tensors not on the right device 
            to get device from a tensor, can use `tensor.device`
    '''

    # Create a tensor
    some_tensor = torch.rand(3, 4)
    print(f"\nsome_tensor: \n{some_tensor}")

    # Find out details about some_tensor
    print(f"\n Datatype: \n{some_tensor.dtype}")
    print(f"\n size: \n{some_tensor.size()}")
    print(f"\n shape: \n{some_tensor.shape}")
    print(f"\n device: \n{some_tensor.device}")


def manipulation_tensors():
    '''
        Tensor operations include:
            * Addition
            * Subtraction
            * Multiplication (element-wise)
            * Dicision
            * Matrix multiplication
    '''

    # Create a tensor and add 100 to it
    tensor = torch.tensor([1,2,3])
    print(f"\ntensor: \n{tensor}")

    # add 100
    tensor_add = tensor + 100
    print(f"\ntensor + 100: \n{tensor_add}")

    # multiply by 10
    tensor_mul = tensor * 10
    print(f"\nmultiply by 10: \n{tensor_mul}")

    # Try out PyTorch in-built functions
    tensor_built_in_add = torch.add(tensor, 10)
    print(f"\nbuilt in add: \n{tensor_built_in_add}")

    tensor_built_in_mul = torch.mul(tensor, 10)
    print(f"\nbuilt in mul: \n{tensor_built_in_mul}")


def matrix_multiplications():
    '''
        Two main ways of performing multiplication in neural networks and deep learuning:
            1. Eleement-wise multiplocation
            2. Matrix multiplication (dot product)

        There are two main rules that performing matrix multiplication need to satisfy:
            1. The **inner dimensions** must match:
            * `(3, 2) @ (3, 2)` won't work
            * `(2, 3) @ (3, 2)` will work
            * `(3, 2) @ (2, 3)` will work
    '''

    tensor = torch.tensor([1,2,3])

    # Element wise multiplication
    print("\nElement wise multiplication:")
    print(f"{tensor} * {tensor}")
    print(f"Equals: \n{tensor * tensor}")

    # Matrix multiplication
    print("\nMatrix multiplication:")
    mat_mul = torch.matmul(tensor, tensor)

    print(f"{tensor} @ {tensor}")
    print(f"Equals: {mat_mul}")

    # Transpose
    tensorA = torch.tensor([[1,2],
                            [3,4],
                            [5,6]])
    
    tensorB = torch.tensor([[3,1],
                            [8,9],
                            [2,7]])
    
    print(f"\ntensor A: \n{tensorA}\n   \ntensor B: \n{tensorB}")
    
    tensorB_T = tensorB.T
    print(f"\ntranspose of tensor B:\n{tensorB_T}")

    mat_mul_T = torch.matmul(tensorA, tensorB_T)
    print(f"\ntensor A @ tensor B.T:\n{mat_mul_T}")


def find_min_max_mean_sum():
    # tensor:
    x = torch.arange(0,100,10)

    # find min
    x_min = torch.min(x)

    # find max
    x_max = torch.max(x)

    # find mean
    x_mean = torch.mean(x.type(torch.float))

    # find sum
    x_sum = torch.sum(x.type(torch.float))

    # find min idx (positional min)
    x_min_idx = torch.argmin(x)
    x_min = x[x_min_idx.item()]

    # find max idx (positional max)
    x_max_idx = torch.argmax(x)
    x_max = x[x_max_idx.item()]


    debug=1


def reshaping_stacking_squeezing():
    '''
        To change tensor shape or dimension:
            Reshaping:
                reshapes an input tensor to a defined shape
            View:
                return a view of an input tensor of certain shape 
                but keep the same memory as the original tensor
            Stacking:
                combine multiple tensors on top of each other(vstack)
                or side by side(hstack)
            Squeezing:
                removes all 1 dimensions from a tensor
            Unsqueezing:
                add a 1 dimension to a target tensor
            Permute:
                return a view of the input with 
                dimensions permuted(swapped) in a certain way
    '''

    # tensor
    x = torch.arange(1., 10.) # tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])

    # add a dimension
    x_reshaped1 = x.reshape(1,9)
    x_reshaped2 = x.reshape(9,1)

    # change the view
    z = x.view(1,9)
    '''
        z share the same view of x, z changes, x follows
    '''

    # stack tensor on top each other
    x_stacked_dim0 = torch.stack([x,x,x], dim=0)
    x_stacked_dim1 = torch.stack([x,x,x], dim=1)

    # squeeze / unsqueeze
    x_unsqueezed = torch.unsqueeze(x, dim=1)
    x_squeezed = torch.squeeze(x)

    # premute: change the view
    x_original = torch.rand(1,2,3)
    x_premuted = x_original.permute(2,0,1)
    # check the shape in order 2,0,1
    x_original_shape = x_original.shape
    x_premuted_shape = x_premuted.shape

    debug=1


def indexing():
    # create a tensor with reshaped
    x = torch.arange(1,10).reshape(1,3,3)   # [123456789] to 3x3

    idx_1 = x[0][0][0]
    idx_3 = x[0][0][2]
    idx_9 = x[0][2][2]
    idx_258 = x[:,:,1]

    debug=1


def numpy():
    array = np.arange(1.0, 8.0) # default dtype is float64
    tensor_fnp = torch.from_numpy(array) # dtype=float64
    tensor_ori = torch.arange(1.0, 8.0) # dtype=float32

    # can set to same dtype as ori
    tensor_f32 = torch.from_numpy(array).type(torch.float32)

    # tensor to numpy
    tensor_np = tensor_f32.numpy()


    debug=1



# fun = pytorch_version()
# fun = scalar_vector_matrix_TENSOR()
# fun = random_tensor()
# fun = zeros_and_ones()
# fun = range_of_tensors_and_tensors_like()
# fun = tensor_datatypes()
# fun = getting_information_from_tensors()
# fun = manipulation_tensors()
# fun = matrix_multiplications()
# fun = find_min_max_mean_sum()
# fun = reshaping_stacking_squeezing()
# fun = indexing()
# fun = numpy()



debug=1