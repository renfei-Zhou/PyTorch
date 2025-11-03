import torch
from torch import nn
import matplotlib.pyplot as plt

# check pytorch version
torch.__version__


# setup device
device = "cuda" if torch.cuda.is_available() else "cpu"



debug=1