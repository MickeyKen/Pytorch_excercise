import torch
from torch import optim
from torch import nn
from torchvision.models import vgg11

# 平均二乗誤差の損失関数
criterion = nn.MSELoss()

x = torch.Tensor([0,1,2])
y = torch.Tensor([1,-1,0])

# xとyの各要素の差の二乗の平均値
loss = criterion(x,y)

print(loss)

# 既存のモデルを使用する→ハイパーパラメータの確認
model = vgg11()
optimizer = optim.Adam(model.parameters())
print(optimizer)

# 演習課題
x2 = torch.Tensor([1,1,1,1])
y2 = torch.Tensor([0,2,4,6])

loss2 = criterion(x2, y2)
print(loss2)