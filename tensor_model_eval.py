import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

# モデル作成の注意点：初期化関数/__init__とforward()の名前は決まっている
class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()

        # 全結合層を2つ定義する
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512,10)

        # 損失関数と活性化関数を定義する
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.parameters(), lr = 0.01)
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        print("[fc1を通過]\n", x)
        x = F.relu(x)
        print("[reluを通過]\n", x)
        x = self.fc2(x)
        return x

def train(model, train_loader):

    # 学習モードを明示する
    model.train()

    total_correct = 0
    total_loss = 0
    total_data_len = 0

    for batch_imgs, batch_labels in train_loader:
        # 28×28の画像に変換する
        batch_imgs = batch_imgs.reshape(-1, 28*28*1)
        # 正解ラベルをone-hot処理
        labels = torch.eye(10)[batch_labels]

        # 順伝搬する
        outputs = model(batch_imgs)
        # 勾配を初期化する
        model.optimizer.zero_grad()
        # 損失を計算する
        loss = model.criterion(outputs, labels)
        # 逆伝搬で勾配を計算する（微分する）
        loss.backward()
        # 最適化する
        model.optimizer.step()

        # 以下は正解率と損失等の学習の進み具合の見える化
        _,pred_labels = torch.max(outputs, axis = 1)
        batch_size = len(batch_labels)

        for i in range (batch_size):
            total_data_len += 1
            if pred_labels[i] ==batch_labels[i]:
                total_correct += 1
        total_loss += loss.item()
    accuracy = total_correct / total_data_len * 100
    loss = total_loss / total_data_len

    return accuracy, loss

def test(model, data_loader):
    # 評価モードにする
    model.eval()

    total_data_len = 0
    total_correct = 0

    for batch_imgs, batch_labels in data_loader:
        outputs = model(batch_imgs.reshape(-1,28*28*1))

        _,pred_labels = torch.max(outputs, axis = 1)
        batch_size = len(pred_labels)
        for i in range(batch_size):
            total_data_len += 1
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1

    acc = 100.0 * total_correct / total_data_len
    return acc



# テストデータの読み込み
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=100,
                                            shuffle=False, 
                                            num_workers=2)

model = mlp_net()

acc, loss = train(model, train_loader)
print(f"正解率: {acc}, 損失: {loss}")

test_acc = test(model, test_loader)
print(test_acc)