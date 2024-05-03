import torch

a = torch.ones(3)
# rand is random
b = torch.rand(3)

print(a)
print(b)
print(a + 3)
print(a * 3)
print(a + b)

c  = torch.ones(3,2)
d = torch.zeros(3,2)
print(c*3 + d*2)