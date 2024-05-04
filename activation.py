import torch
from torch.nn import functional as F

x = torch.tensor([-2.0, 1.0, 4.0, 0.0])

print('relu: ', F.relu(x))
print('softmax: ', F.softmax(x, dim = 0))
print('sigmoid: ', torch.sigmoid(x)) 

# 演習課題
x2 = torch.tensor([-5.0, 5.0, -10.0, 10.0])

print('relu: ', F.relu(x2))
print('softmax: ', F.softmax(x2, dim = 0))
print('sigmoid: ', torch.sigmoid(x2)) 

