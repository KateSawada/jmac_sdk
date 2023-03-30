import os
import numpy as np
from mjx import Observation, State, Action 

if __name__ == "__main__":
    '''
    1000000個ごとに１つのファイルとして保存する
    そうしないと、メモリが足らん
    '''

    # 天鳳のログのパス
    dir_path = './json'

    # obsのデータ
    obs_hist = []
    # 取った行動のデータ
    action_hist = []

    file_num = 0
    files = os.listdir(dir_path)

    for file in files:
        # ファイルを開いてデータを読み込む
        with open(os.path.join(dir_path, file)) as f:
            lines = f.readlines()
            for line in lines:
                state = State(line)
                for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
                    # 盤面の情報
                    obs = Observation._from_cpp_obj(cpp_obs)
                    feature = obs.to_features(feature_name="mjx-small-v0")
                    # 行動
                    action = Action._from_cpp_obj(cpp_act)
                    action_idx = action.to_idx()
                    # 格納
                    obs_hist.append(feature.ravel())
                    action_hist.append(action.to_idx())
        
        # 1000000件超えたら保存
        if np.array(action_hist).shape[0] >= 1_000_000:
            # ファイルの個数を出力しておく
            print(file_num, len(action_hist))
            # 小分けで保存する必要がある
            np.save("./data/tenho_obs_"+str(file_num)+".npy", np.array(obs_hist, dtype=np.int32))
            np.save("./data/tenho_actions_"+str(file_num)+".npy", np.array(action_hist, dtype=np.int32))

            # リセット
            obs_hist = []
            action_hist = []
            # ファイル番号
            file_num += 1

    # 残りをファイルに書き出し
    np.save("./data/tenho_obs_"+str(file_num)+".npy", np.array(obs_hist, dtype=np.int32))
    np.save("./data/tenho_actions_"+str(file_num)+".npy", np.array(action_hist, dtype=np.int32))