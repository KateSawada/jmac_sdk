import glob
import os
import argparse

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
    jsons = jsons[int(args.start * len(jsons)): int(args.end * len(jsons))]

    print(f"started {len(jsons)} file ({args.start} : {args.end})")

    for file in jsons:
        os.makedirs(os.path.join(args.output, os.path.basename(file)), exist_ok=True)

        with open(file) as f:
            rounds = 0
            lines = f.readlines()

            for line in lines:  # line means round
                round_path = os.path.join(args.output, os.path.basename(file), str(rounds))
                os.makedirs(round_path, exist_ok=True)
                for p in range(4):  # make directories for each players
                    os.makedirs(os.path.join(round_path, str(p)), exist_ok=True)

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

                state = State(line)

                for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
                    obs = Observation._from_cpp_obj(cpp_obs)
                    feature = obs.to_features(feature_name="mjx-large-v0")

                    action = Action._from_cpp_obj(cpp_act)
                    action_idx = action.to_idx()
                    # if (action.type() == ActionType.RON):
                    #     print(action.to_json())

                    # print(obs.who())  # int 0 to 3
                    # print(feature)
                    # print(feature.shape)  # (132, 34)

                    # add actual decision as new row to {feature}
                    # feature[132, 0] : index of mjx.ActionType
                    # DISCARD 0
                    # TSUMOGIRI 1
                    # RIICHI 2
                    # CLOSED
                    # ADDED
                    # TSUMO
                    # ABORTIVE_DRAW_NINE_TERMINALS
                    # CHI
                    # PON
                    # OPEN
                    # RON 10
                    # PASS 11
                    # DUMMY 99
                    # feature[132, 1] : tile or open if it described
                    actual_action = np.zeros((1, 34))
                    actual_action[0, 0] = int(action.type())
                    if action.open():
                        actual_action[0, 1] = action.open().bit
                    elif action.tile():
                        actual_action[0, 1] = action.tile().id()
                    feature = np.concatenate((feature, actual_action), axis = 0)
                    np.save(os.path.join(round_path, str(obs.who()), f"{action_counts[obs.who()]}.npy"), feature)  # NOTE: comment out for debug

                    action_counts[obs.who()] += 1
                rounds += 1

if __name__ == "__main__":
    main()
