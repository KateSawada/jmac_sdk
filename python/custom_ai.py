import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mjx import Action, Observation, State
from pytorch_lightning import Trainer
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, TensorDataset


def make_dataset():
    dir_path = "json"
    batch_size = 10  # バッチサイズの設定
    data_set = []

    files = glob.glob("./datas/*")
    files = files[:1300]
    num_files = len(files)

    for i in range(0, num_files, batch_size):
        batch_files = files[i:i+batch_size]

        batch_data = []
        for file in batch_files:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    state = State(line)
                    for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
                        obs = Observation._from_cpp_obj(cpp_obs)
                        # feature = obs.to_features(feature_name="mjx-large-v0").ravel()
                        feature = obs.to_features(feature_name="mjx-small-v0").ravel()

                        action = Action._from_cpp_obj(cpp_act)
                        action_idx = action.to_idx()
                        data = np.append(feature, action_idx)
                        batch_data.append(data)

        batch_data = np.array(batch_data)
        data_set.append(batch_data)

    data_set = np.concatenate(data_set)
    np.save("dataset/small-1300", data_set)


def int_to_binary_vector(num):
    binary = bin(num)[2:].zfill(8)  # 8桁の2進数表現を取得
    vector = np.array(list(binary)).astype(int)  # 1次元配列に変換
    return vector


def binary_vector_to_int(vector):
    binary_str = ''.join(vector.astype(str))  # 1次元配列を文字列に変換
    num = int(binary_str, 2)  # 2進数表現の文字列を10進数の整数値に変換
    return num


# 学習データに対する処理
class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results


# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'val_loss': loss}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        results = {'val_loss': avg_loss}
        return results


# テストデータに対する処理
class TestNet(pl.LightningModule):

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'test_loss': loss}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'test_loss': avg_loss}
        return results


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):

    def __init__(self, input_size=544, hidden_size=544, output_size=181, batch_size=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    # New: 平均ニ乗誤差
    def lossfun(self, y, t):
        return F.mse_loss(y, t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return torch.optim.SGD(self.parameters(), lr=0.1)
        return optimizer


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net2(TrainNet, ValidationNet, TestNet):

    def __init__(self, input_size=544, hidden_size=544, output_size=181, batch_size=100):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 362)
        self.fc3 = nn.Linear(362, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    # New: 平均ニ乗誤差
    def lossfun(self, y, t):
        return F.mse_loss(y, t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return torch.optim.SGD(self.parameters(), lr=0.1)
        return optimizer


if __name__ == "__main__":
    # make_dataset()
    dataset = np.load("dataset/small-1300.npy")
    x = dataset[:, :-1]
    y = dataset[:, -1]

    y = np.eye(181)[y]   # one hot表現に変換

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, y)

    n_train = int(len(dataset) * 0.7)
    n_val = int(len(dataset) * 0.2)
    n_test = len(dataset) - n_train - n_val

    # ランダムに分割を行うため、シードを固定して再現性を確保
    torch.manual_seed(0)

    # データセットの分割
    train, val, test = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test])

    torch.manual_seed(0)

    # インスタンス化
    net = Net2()
    trainer = Trainer(max_epochs=30)

    # 学習の実行
    trainer.fit(net)

    trainer.test()

    # torch.save(net, 'models/second_model.pth')
    # model = torch.load("models/second_model.pth")
    torch.save(net.state_dict(), 'models/fifth_model.pth')
