import torch 
import numpy as np

num1 = np.zeros((2,3))
ten1 = torch.zeros((2,3))

# dtype = shape of array
# dtype = shape of elements of array
print(f"num1: {num1.shape} / {num1.dtype} ")

print(f"ten1: {ten1.shape} / {ten1.dtype} ")

print(type(ten1))

# default value / numpy int32,float64 / pytorch int64,float32
ten2 = torch.ones((2,3), dtype = torch.int32)
print(ten2)

# convert from toroch to numpy
# caution:: Converted value is sharing memory each other. 
before = torch.zeros(3)
print(f"変換前: {before}")
after = before.numpy()
print(f"変換後: {after}")
data = torch.from_numpy(after)
print(f"再変換後: {data}")

pro21 = np.ones((2,3))
pro22 = torch.from_numpy(pro21)
print(f"変換後{pro22}")