# Author: Julian Haubold
# https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=819s&ab_channel=freeCodeCamp.org bis zu min 35
import torch
import numpy as np

print("______ Different tensors")
# Number
t1 = torch.tensor(4.)
print(t1)
print(t1.dtype)
# Vector
t2 = torch.tensor([1.,2,3,4,5,6,7,8,9,10])
print(t2)
# Matrix
t3 = torch.tensor([[3.,4],[5.,6],[7.,8],[9.,10]])
print(t3)
# 3-dimensional array
t4 = torch.tensor([
    [[11,12,13],
     [13,14,15]],
    [[15,16,17],
     [17,18,19.]]])
print(t4)

print("\n______ Types of these tensors")
# shapes
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)


print("\n\n______ Tensor operations and gradients")
# Create Tensors
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

# Arithmetic operations
y = w * x + b
y.backward()

# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

print("\n\n______ Interoperability with Numpy")
x = np.array([[1,2],[3,4],[5,6]])
print (x)
y = torch.from_numpy(x)
print(y)