import random

import mjx
from mjx.const import ActionType
import numpy as np

from client.agent import CustomAgentBase

class RuleAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.toitoi_toitsu = 3  # 3対子あったら鳴きまくって対々和狙う

    def judge_toitoi(self, features: np.ndarray, obs: mjx.Observation):
        """対々和を目指すか判断して行動

        Args:
            features (np.ndarray): features array
            obs (mjx.Observation): observation
        """
        legal_actions = obs.legal_actions()
        hand = features[0:4]
        discarded_over_two_tiles = np.where(np.sum(features[13:25], axis=0) >= 2)[0]
        n_toitsu_shuntsu = np.count_nonzero(np.sum(hand, axis=0) >= 2)
        if n_toitsu_shuntsu >= self.toitoi_toitsu:
            # 対々和狙う動き
            act_candidate = []  # ツモに対する動き
            hand_over_two_tiles = np.where(np.sum(hand, axis=0))[0]
            for i in range(len(legal_actions)):
                # ポンできるとき，その牌が手牌に3枚未満だったらポン
                if (legal_actions[i].type() == ActionType.PON and np.sum(hand, axis=0)[legal_actions[i].open().tiles_from_hand()[0].id() // 4] < 3):
                    return legal_actions[i]
                # カンできるときはする
                elif (legal_actions[i].type() in [ActionType.OPEN_KAN, ActionType.CLOSED_KAN, ActionType.ADDED_KAN]):
                    return legal_actions[i]
                elif (legal_actions[i].type() in [ActionType.TSUMOGIRI, ActionType.DISCARD]):
                    if (legal_actions[i].tile().id() // 4 in hand_over_two_tiles and legal_actions[i].tile().id() // 4 not in discarded_over_two_tiles):
                        act_candidate.append(legal_actions[i])
            if len(act_candidate) > 0:
                return random.choice(act_candidate)

        else:
            return None


    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        legal_actions =obs.legal_actions()
        features = obs.to_features("mjx-large-v0")
        for i in range(len(legal_actions)):
            # リーチ・和了できる時はそれ
            if (legal_actions[i].type() in [ActionType.RIICHI, ActionType.TSUMO, ActionType.RON]):
                return legal_actions[i]

        # 対々和狙うか判断
        toitoi = self.judge_toitoi(features, obs)
        if toitoi is not None:
            return toitoi

        is_nakinashi = False

        for i in range(len(legal_actions)):
            # 鳴き無し
            if (not is_nakinashi):
                if (legal_actions[i].type() in [ActionType.CHI, ActionType.OPEN_KAN, ActionType.PON]):
                    print(legal_actions[i].type())
                    print(legal_actions[i].open())
                    # print([legal_actions[i].open().tiles()[i] for i in range(len(legal_actions[i].open().tiles()))])
                    print(legal_actions[i].open().tiles_from_hand()[0].id(), legal_actions[i].open().tiles_from_hand()[1].id())
                    print(legal_actions[i].open().stolen_tile().id())
                    # input()
                    for j in range(len(legal_actions)):
                        if legal_actions[j].type() == ActionType.PASS:
                            return legal_actions[j]




        print("shanten")
        print(features[73:80])
        print("hand")
        print(features[0:7])
        print("effective discard")
        print(features[80])
        effective_discards = np.where(features[80] == 1)[0]
        print()
        for i in range(len(legal_actions)):
            if (legal_actions[i].type() in [ActionType.DISCARD, ActionType.TSUMOGIRI] and legal_actions[i].type() not in [ActionType.RIICHI, ActionType.ADDED_KAN, ActionType.CLOSED_KAN, ActionType.OPEN_KAN, ActionType.ABORTIVE_DRAW_NINE_TERMINALS]):
                if (legal_actions[i].tile().id() // 4 in effective_discards):
                    return legal_actions[i]
        return random.choice(obs.legal_actions())
