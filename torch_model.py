import torch
from torch import nn
from torch.nn import functional as F

# 入力層が4 出力層が2　の　全結合層(Linear)
fc = nn.Linear(4,2)
x = torch.Tensor([1,2,3,4])

x = fc(x)
print(x)

# 活性化関数
x = torch.Tensor([-1.0, -0.5, 0.5, 1.0])
x = F.relu(x)
print(x)


# 演習課題
fc2 = nn.Linear(4,2)
x2 = torch.Tensor([-2,-1,1,2])
# 順伝搬
x2 = fc2(x2)
print(x2)
# 活性化関数処理
x2 = F.relu(x2)
print(x2)