# Author: Julian Haubold
# https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=819s&ab_channel=freeCodeCamp.org bis zu min 19
import torch

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
