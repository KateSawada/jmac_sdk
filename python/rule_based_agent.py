import random

import mjx
from mjx.const import ActionType
import numpy as np

from client.agent import CustomAgentBase

class RuleAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.have_yakuhai = False
        self.is_print = False
        np.set_printoptions(threshold=np.inf)

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        legal_actions =obs.legal_actions()
        features = obs.to_features("mjx-large-v0")
        hand_count = np.sum(features[0:4], axis=0)
        # 局の切り替わり時に役の情報をリセット
        if np.sum(features[13:16]) == 0:
            # print("new round")
            self.have_yakuhai = False
        # if self.is_print:
        #     print("3 ===")
        #     print(hand_count)
        #     print(features[37 :41])
        #     self.is_print = False
        is_pon = False
        pon_tile = -1
        pon_act = None
        is_cii = False
        cii_act = None
        is_tsumo = False

        for i in range(len(legal_actions)):
            # リーチ・和了できる時はそれ
            if (legal_actions[i].type() in [ActionType.RIICHI, ActionType.TSUMO, ActionType.RON]):
                return legal_actions[i]
            elif (legal_actions[i].type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]):
                is_tsumo = True
            elif (legal_actions[i].type() in [ActionType.PON]):
                is_pon = True
                pon_tile = legal_actions[i].open().stolen_tile().id() // 4
                pon_act = legal_actions[i]
            elif (legal_actions[i].type() in [ActionType.CHI]):
                is_cii = True
                cii_act = legal_actions[i]

        shown_count = np.sum(features[13:25], axis=0)  # discarded => 13:25, discarded_from_hand => 25:37, opened => 37: 65
        shown_count += np.sum(features[37:41], axis=0)
        shown_count += np.sum(features[44:48], axis=0)
        shown_count += np.sum(features[51:55], axis=0)
        shown_count += np.sum(features[58:62], axis=0)
        shown_count += np.sum(features[69: 73], axis=0)  # dora

        # 鳴かれた牌は捨てた牌と副露で2重にカウントされちゃうから苦肉の策…
        shown_count = np.where(shown_count > 4, 4, shown_count)


        # 役にならない字牌が3枚あったらそのまま刻子として使う
        exclude = []
        if len(np.where(hand_count[27:] == 3)[0]) != 0:
            exclude += list(np.where(hand_count[27:] == 3)[0])

        # 自風・場風以外の風牌は切る
        targets = [27, 28, 29, 30]
        targets.remove(27 + (int(legal_actions[i].who()) - obs.round()) % 4)
        # if (27 + int(legal_actions[i].who()) - obs.round()) % 4 in targets:
        if (27 + obs.round() // 4 in targets):
            targets.remove(27 + obs.round() // 4)
        if np.any(hand_count[targets]) > 0:
            # print(legal_actions[i].who(), int(legal_actions[i].who()), obs.round())
            # print(int(legal_actions[i].who()))
            # print(27 + (int(legal_actions[i].who()) - obs.round()) % 4)
            # print(targets)
            # print(hand_count)
            for i in range(len(legal_actions)):
                if (legal_actions[i].tile() is not None and legal_actions[i].tile().id() // 4 in targets):
                    if (not legal_actions[i].tile().id() // 4 in exclude):
                        return legal_actions[i]

        # 自風場風・三元牌が手牌に2枚あったら鳴く
        if is_pon and (pon_tile in [31, 32, 33, 27 + obs.round() // 4, 27 + (int(legal_actions[i].who()) - obs.round()) % 4] and hand_count[pon_tile] == 2):
            # print([31, 32, 33, 27 + obs.round() // 4, 27 + (int(legal_actions[i].who()) - obs.round()) % 4])
            # print(pon_tile)
            self.have_yakuhai = True
            return pon_act
        # (player - round) % 4 = 風
        # print(int(legal_actions[i].who()))

        # N巡目以降は1枚だけの字牌→2枚だけの字牌の順に切る
        # 4枚あったら暗槓
        # discardの1がいくつあるかで順目を判断
        zihai_discard_zyunme = 6
        if (np.sum(features[13:16]) > zihai_discard_zyunme and (np.any(hand_count[27:] >= 1))):
            # 4枚あったら暗槓
            if len(np.where(hand_count[27:] == 4)[0]) != 0:
                target = np.where(hand_count[27:] == 4)[0][0]
            # 1枚しか持っていないものから切る
            else:
                for i in range(len(legal_actions)):
                    if (legal_actions[i].tile() is not None and legal_actions[i].tile().id() // 4 in [27, 28 ,29, 30, 31, 32, 33]):
                        if (hand_count[legal_actions[i].tile().id() // 4] == 1):
                            return legal_actions[i]
                for i in range(len(legal_actions)):
                    if (legal_actions[i].tile() is not None and legal_actions[i].tile().id() // 4 in [27, 28 ,29, 30, 31, 32, 33]):
                        if (hand_count[legal_actions[i].tile().id() // 4] == 2):
                            return legal_actions[i]

        # 字牌1枚見えていたら切る
        if (is_tsumo and np.any((shown_count[27:] >= 1) & (hand_count[27:] >= 1))):
            target_tile = np.where((shown_count[27:] >= 1) & (hand_count[27:] >= 1))[0][0] + 27

            for i in range(len(legal_actions)):
                if (legal_actions[i].tile() is not None and legal_actions[i].tile().id() // 4 == target_tile):
                    return legal_actions[i]

        # 役があるかチェック
        if np.any(hand_count[[31, 32, 33, 27 + obs.round() // 4, 27 + (int(legal_actions[i].who()) - obs.round()) % 4]] >= 3):
            # print("2 ===")
            # print(hand_count)
            self.have_yakuhai = True

        # 役があれば積極的に鳴く
        if (is_pon or is_cii):
            if self.have_yakuhai:
                # print("1 ===")
                # print(legal_actions[i].who())
                # print(obs.round())
                # print(hand_count)
                # print(features[37:41])
                if pon_act:
                    return pon_act
                elif cii_act:
                    return cii_act
            else:
                for i in range(len(legal_actions)):
                    if (legal_actions[i].type() == ActionType.PASS):
                        return legal_actions[i]
            raise ValueError("Invalid action")

        # 最終的に，effectiveな判断
        effective_discard = features[80]
        for i in reversed(range(len(legal_actions))):
            if (legal_actions[i].tile() is not None and legal_actions[i].tile().id() // 4 in effective_discard):
                return legal_actions[i]
        return legal_actions[0]


