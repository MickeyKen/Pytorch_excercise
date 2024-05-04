from torch import nn
import torch

x = torch.tensor([[0.2, 0.5, 0.3]])
mse_label = torch.tensor([[0, 1, 0]])
cel_label = torch.tensor([1])

# (予測値, 正解値)を投げる
print('MSE: ', nn.MSELoss(reduction='mean')(x, mse_label))
print('CrossEntropy: ', nn.CrossEntropyLoss()(x, cel_label))


# 演習課題
x2 = torch.tensor([[0.3, 0.3, 0.3, 0.1]])
mse_label2 = torch.tensor([[0, 0, 0, 1]])
cel_label2 = torch.tensor([3])

# (予測値, 正解値)を投げる
print('MSE: ', nn.MSELoss(reduction='mean')(x2, mse_label2))
print('CrossEntropy: ', nn.CrossEntropyLoss()(x2, cel_label2))