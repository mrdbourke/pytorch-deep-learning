
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
# print(torch.__version__)

# multiply two tensors
x = torch.tensor(3.)    # create a tensor with a single number
w = torch.tensor(4., requires_grad=True)    # create a tensor with a single number  # requires_grad=True indicates that we want to compute gradients with respect to these tensors during backward pass

# Arithmetic operations
y = w * x
print(y)
