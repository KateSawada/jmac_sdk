import glob
import os
import argparse

import tqdm
import json
import numpy as np
from mjx import Observation, State, Action, ActionType


N_YAKU = 55

def main():
    print(os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-s", "--start", type=float, default=0.)
    parser.add_argument("-e", "--end", type=float, default=1.)

    args = parser.parse_args()

    jsons = glob.glob(os.path.join(args.input, "*"))

    bar = tqdm.tqdm(total = len(jsons))

    for file in jsons:
        os.makedirs(os.path.join(args.output, os.path.basename(file)), exist_ok=True)

        with open(file) as f:
            rounds = 0
            lines = f.readlines()

            for line in lines:  # line means round
                round_path = os.path.join(args.output, os.path.basename(file), str(rounds))
                os.makedirs(round_path, exist_ok=True)

                action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # number of actions for each players
                json_obj =  json.loads(line)

                # wins_info
                wins_npy = np.zeros((4, N_YAKU))

                # when someone(s) won
                if 'wins' in json_obj["roundTerminal"].keys():
                    wins_info = json_obj["roundTerminal"]["wins"]
                    for i in range(len(wins_info)):
                        if not "who" in wins_info[i].keys():
                            who = 0
                        else:
                            who = wins_info[i]["who"]
                        if ("yakus" in wins_info[i].keys()):
                            for yaku_idx in wins_info[i]["yakus"]:
                                wins_npy[who, yaku_idx] = 1
                        if ("yakusmans" in wins_info[i].keys()):
                            for yaku_idx in wins_info[i]["yakumans"]:
                                wins_npy[who, yaku_idx] = 1
                np.save(os.path.join(round_path, "wins.npy"), wins_npy)  # NOTE: comment out for debug

                with open(os.path.join(round_path, "state.json"), "w") as f:
                    f.write(line)
                rounds += 1
        bar.update(1)

if __name__ == "__main__":
    main()
