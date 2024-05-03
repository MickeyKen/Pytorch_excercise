import torch

# 4行3列
x = torch.rand((4,3))

print(x)

# 転置するときは(0,1)で指定する　→ torch.t(tensor名)も同じ
print(x.transpose(0,1))

x = torch.ones(16)

print(x)
print(x.reshape(2,8))
# -1指定で自動で形状を変更してくれる
print(x.reshape(4,-1))

x = torch.tensor([[1,1,1] , [1,2,2] , [2,2,3] , [3,3,3]])
print(x.reshape(3,-1))

# zero_ は破壊的メソッドである(インプレース)
# pytorchにおいては、_ (アンダーバー)は破壊的メソッドであることをしめす
a = torch.tensor([[1,2,3], [4,5,6]])
print(a)
a.zero_()
print(a)