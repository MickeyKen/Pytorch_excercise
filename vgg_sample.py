from torchvision import models
from torch import nn
import torch

class vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg11() # pretrained=Trueですでに学習済のモデルを使用できる
        # デフォルトは1000ラベルなので、10ラベルにしたい場合
        self.fc = nn.Linear(1000, 10)


    def forward(self, x):
        x = self.vgg(x)
        x = self.fc(x)
        return x

model1 = vgg()
print(model1)