from torch import nn
import torch

# 入力が3チャンネル/出力が5チャンネル/カーネルが3の定義
conv = nn.Conv2d(3, 5, 3)

# rand(1,3,28,28)で3チャンネル(カラー)の画像データを生成できる
x = torch.Tensor(torch.rand(1,3,28,28))
x = conv(x)

print(x.shape)

# 演習課題
conv2 = nn.Conv2d(1, 2, 3)

# 100枚の画像データを生成する
x2 = torch.Tensor(torch.rand(100,1,64,64))
x2 = conv2(x2)

print(x2.shape)