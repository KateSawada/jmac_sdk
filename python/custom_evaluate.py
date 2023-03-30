import json
import os

import numpy as np


def evaluate_from_nall_logs():
    # ディレクトリのパスを指定
    dir_path = "logs"

    dir_list = os.listdir(dir_path)

    # print(dir_list)

    dir_paths = [os.path.join(dir_path, dir_name, "player_0.json")
                 for dir_name in dir_list]
    final_tens = []

    for dir_path in dir_paths:
        with open(dir_path, mode="r") as f:
            d = json.load(f)
            who = d.get("who", 0)
            final_tens.append(d["roundTerminal"]["finalScore"]["tens"][who])
    print(final_tens)
    print(np.array(final_tens).mean())


def evaluate_from_arg(dir_list):

    dir_paths = [os.path.join(dir_path, "player_0.json")
                 for dir_path in dir_list]
    final_tens = []

    for dir_path in dir_paths:
        with open(dir_path, mode="r") as f:
            d = json.load(f)
            who = d.get("who", 0)
            final_tens.append(d["roundTerminal"]["finalScore"]["tens"][who])
    print(final_tens)
    print(np.array(final_tens).mean())
