import torch
from torch import optim
from torchvision.models import vgg11

model = vgg11()

adam = optim.Adam(model.parameters())
sgd = optim.SGD(model.parameters() , lr = 0.01)
momentum = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

print('adam: ', adam)
print('sgd: ', sgd)
print('momentum: ', momentum)