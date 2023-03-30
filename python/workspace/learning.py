import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import time


from model import MLP, CNN_MLP2, CNN_MLP3

if __name__ == "__main__":

    # ニューラルネットワークのインスタンスを生成
    model = CNN_MLP3()


    # 最適化手法の選択
    optimizer = model.configure_optimizers()

    # 検証用のデータ
    test_tgts = np.load('./data/tenho_actions_49.npy')
    test_inps = np.load('./data/tenho_obs_49.npy')

    test_dataset = TensorDataset(torch.Tensor(test_inps[0:100000]), torch.LongTensor(test_tgts[0:100000]))
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # 検証
    print('\n==================================================================\n')
    accuracy = 0
    for batch in test_loader:
        test_inp, test_tgt = batch
        accuracy += int(sum(torch.argmax(model.forward(test_inp), dim=1) == test_tgt))
    print('i: {}, time: {},  accuracy: {}/100000.'.format(0, 0.0, accuracy))


    # エポック数
    MAX_EPOCHS = 5

    # データファイルの数
    DATA_FILES = 49

    for epoch in range(MAX_EPOCHS):
        print('\n============================= epoch '+str(epoch)+' ============================\n')
        for i in range(DATA_FILES):
            # 学習用のデータの読み込み
            tgts = np.load('./data/tenho_actions_'+str(i)+'.npy')
            inps = np.load('./data/tenho_obs_'+str(i)+'.npy')

            # データセットを作成
            dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
            train_loader = DataLoader(dataset, batch_size=1024)


            # 時間計測用
            start_time = time.time()
            for batch in train_loader:
                # バッチサイズ分のサンプルの抽出
                x, t = batch
                # パラメータの勾配を初期化
                optimizer.zero_grad()

                # 予測値の算出
                y = model.forward(x)

                # 目標値と予測値から目的関数の値を算出
                loss = model.loss_module(y, t)

                # 目的関数の値を表示して確認
                # print(loss)
                
                # 各パラメータの勾配を算出
                loss.backward()

                # 勾配の情報を用いたパラメータの更新
                optimizer.step()

            # 検証
            accuracy = 0
            for batch in test_loader:
                test_inp, test_tgt = batch
                accuracy += int(sum(torch.argmax(model.forward(test_inp), dim=1) == test_tgt))
            print('i: {}, time: {},  accuracy: {}/100000.'.format(i, time.time()-start_time, accuracy))

            # メモリ解放
            del inps, tgts, dataset, train_loader

        torch.save(model.state_dict(), "./params/CNN_MLP3/model_"+str(epoch)+".pth")

