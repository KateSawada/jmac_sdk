import random

import mjx
import sys
from datetime import datetime

from client.agent import CustomAgentBase

class ChonAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        #print(obs.draws()[-1].type())
        legal_actions = obs.legal_actions()
        if len(legal_actions) == 1:
            #print("なんもしん")
            return legal_actions[0]

        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [mjx.ActionType.TSUMO, mjx.ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            #print("かちました")
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == mjx.ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            #print("立直")
            return riichi_actions[0]

        # if it can apply chi/pon/open-kan, choose randomly
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [mjx.ActionType.PON, mjx.ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            if a.tile().type() == 31 or a.tile().type() == 32 or a.tile().type() == 33:
                #print("ぽんち")
                return random.choice(steal_actions)


        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]
        ]

        moji_discards = [a for a in legal_discards if a.tile().num() == None]

        if len(moji_discards) > 0:
            for discard in moji_discards:
                dis_tile = discard.tile()
                count = 0
                for tile in obs.curr_hand().closed_tiles():
                    if dis_tile.type() == tile.type():
                        count = count + 1
                if count <= 1:
                    #print("1文字ツモ")
                    return discard
            for discard in moji_discards:
                dis_tile = discard.tile()
                count = 0
                for tile in obs.curr_hand().closed_tiles():
                    if dis_tile.type() == tile.type():
                        count = count + 1
                if count <= 2:
                    #print("2文字ツモ")
                    return discard

        effective_discard_types = obs.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]

        if len(effective_discards) == 0:
            tsumogiri_action = [a for a in legal_actions if a.type() == mjx.ActionType.TSUMOGIRI]
            #print("ツモぎり")
            return tsumogiri_action[0]

        more_effective_discards = []
        for discard in effective_discards:
            find = False
            for tile in obs.curr_hand().closed_tiles():
                if discard.tile().id() == tile.id():
                    continue
                if discard.tile().type()-1 <= tile.type() <= discard.tile().type()+1 and tile.num() + discard.tile().num() != 10:
                    find = True
                    break
            if find == False:
                more_effective_discards.append(discard)

        #print([a.tile().type() for a in effective_discards])
        #print([a.tile().type() for a in more_effective_discards])

        use_discards = [a for a in more_effective_discards if not a.tile().is_red()]
        if len(use_discards) == 0:
            use_discards = more_effective_discards

        discard_1or9 = [a for a in use_discards if a.tile().num() == 1 or a.tile().num() == 9]
        if len(discard_1or9) > 0:
            #print("1or9ぎり")
            return discard_1or9[0]

        if len(use_discards) > 0:
            #print("ドラなし効果切り")
            return use_discards[0]

        if len(more_effective_discards) > 0:
            #print("ドラあり効果切り")
            return more_effective_discards[0]

        discard_1or9 = [a for a in effective_discards if a.tile().num() == 1 or a.tile().num() == 9]
        if len(discard_1or9) > 0:
            #print("あきらめ1or9ぎり")
            return discard_1or9[0]

        #print("あきらめ切り")
        return effective_discards[0]

        
        