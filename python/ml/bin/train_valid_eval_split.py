import os
import glob
import random
from collections import defaultdict

import hydra
from tqdm import tqdm
import numpy as np

def dahai(input_dir, output_dir):
    pass

def yaku(input_dir, output_dir, ignore, rate):
    all_rounds = glob.glob(os.path.join(os.path.abspath(input_dir), "*", "*"))
    # all_rounds = all_rounds[:100]
    print(f"total rounds: {len(all_rounds)}")

    won_yaku_counts = defaultdict(list)

    data = []
    for round_dir in tqdm(all_rounds, desc="phase 1"):
        wins = np.load(os.path.join(round_dir, "wins.npy"))
        for p in range(4):
            if not np.all(wins[p] == 0):
                won_yakus = np.where(wins[p] > 0)[0]
                for i in range(len(won_yakus)):
                    won_yaku_counts[int(won_yakus[i])] += [round_dir]

    for i in ignore:
        won_yaku_counts[i] = []
    min_yaku_count = len(won_yaku_counts[1])  # 1 is 立直

    for yaku_idx in tqdm(won_yaku_counts.keys(), desc="phase 2"):
        if len(won_yaku_counts[yaku_idx]) > 0:
            if min_yaku_count > len(won_yaku_counts[yaku_idx]):
                min_yaku_count = len(won_yaku_counts[yaku_idx])
            print(f"{yaku_idx} : {len(won_yaku_counts[yaku_idx])}")
    print(f"least count: {min_yaku_count}")

    for yaku_idx in tqdm(won_yaku_counts.keys(), desc="phase 3"):
        if len(won_yaku_counts[yaku_idx]) > 0:
            yaku_data = won_yaku_counts[yaku_idx]
            random.shuffle(yaku_data)
            data += yaku_data[:min_yaku_count]

    print(f"valid data: {len(data)}")

    random.shuffle(data)

    p1 = int(len(data) * rate[0])
    p2 = p1 + int(len(data) * rate[1])
    trains = data[:p1]
    valids = data[p1:p2]
    evals = data[p2:]

    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for file in trains:
            f.write(f"{file}\n")
    with open(os.path.join(output_dir, "valid.txt"), "w") as f:
        for file in valids:
            f.write(f"{file}\n")
    with open(os.path.join(output_dir, "eval.txt"), "w") as f:
        for file in evals:
            f.write(f"{file}\n")


@hydra.main(config_path="config", config_name="train_valid_eval_split")
def main(config):

    assert config.data.type in ("yaku", "dahai")

    if config.data.type == "yaku":
        yaku(config.input, config.data.dir, config.data.ignore, config.data.rate)
    elif config.type == "dahai":
        dahai(config.input, config.data.output_dir)

if __name__ == "__main__":
    main()
