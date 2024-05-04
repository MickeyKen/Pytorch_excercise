import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

# モデル作成の注意点：初期化関数/__init__とforward()の名前は決まっている
class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()

        # 全結合層を2つ定義する
        self.fc1 = nn.Linear(3,5)
        self.fc2 = nn.Linear(5,2)

        # 損失関数と活性化関数を定義する
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        print("[fc1を通過]\n", x)
        x = F.relu(x)
        print("[reluを通過]\n", x)
        x = self.fc2(x)
        return x

model = mlp_net()
x = torch.Tensor([0,1,2])

output = model(x)

print("[モデルの出力]\n", output)

# モデルの中身を確認する
print(mlp_net())
print(mlp_net().optimizer)