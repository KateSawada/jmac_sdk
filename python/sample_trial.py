"""対局を指定した回数行うスクリプト．対局結果の出力も可能．
"""

import argparse
import os
from datetime import datetime
import json
import random
import copy

import mjx
import mjx.agents

from mjx.action import Action
from mjx.const import ActionType
from mjx.env import MjxEnv
from mjx.observation import Observation
from mjx.visualizer.selector import Selector
from mjx.const import EventType, PlayerIdx, TileType
from mjx.event import Event

from discard import *
from search_suzi import *
from check_ten import is_possible_to_rank_up,diff_ten
from action_type_def import ActionModeType,TargetYakuType

from server import convert_log
from client.agent import CustomAgentBase


# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．


class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.remaining_tiles = [[4 for _ in range(34)],[1,1,1]]
        self.action_mode = ActionModeType.MENZEN
        self.target_yaku = TargetYakuType.NO_TARGET
        self.remaining_tiles_num = 70
        self.when_riichi = [-1,-1,-1]
        self.before_riichi_discards_list = [[0 for _ in range(34)] for _ in range(3)]
        self.my_player_id = -1
        self.ankan_num = 0
        self.ankan_types = []
    
    
    def act(self, obs: mjx.Observation) -> mjx.Action:                    
        hand = obs.MjxLargeV0().current_hand(obs)
        riichi = obs.MjxLargeV0().under_riichis(obs)
        discarded_tiles = obs.MjxLargeV0().discarded_tiles(obs)
        opend_tiles = obs.MjxLargeV0().opened_tiles(obs)
        dealer = obs.MjxLargeV0().dealer(obs)
        ranking = obs.MjxLargeV0().rankings(obs)
        effective_draw = obs.MjxLargeV0().effective_draws(obs)[0]
        shanten = obs.MjxLargeV0().shanten(obs)
        honba = obs.MjxLargeV0().honba(obs)
        kyotaku = obs.MjxLargeV0().kyotaku(obs)
        round = obs.MjxLargeV0().round(obs)
        doras = obs.doras()
        dora_in_hand = obs.MjxLargeV0().dora_num_in_hand(obs)
        tens = obs.tens()
        dora_total_num = 0
        dealer_num = -1
        my_rank = 1
        simotya_rank = 1
        toimen_rank = 1
        kamitya_rank = 1
        kyotaku_num = 0
        honba_num = 0
        is_discarded = False
        is_reset = False

        manzu = [0,1,2,3,4,5,6,7,8]
        pinzu = [9,10,11,12,13,14,15,16,17]
        sozu = [18,19,20,21,22,23,24,25,26]
        tyutyan = [1,2,3,4,5,6,7,10,11,12,13,14,15,16,19,20,21,22,23,24,25]
        yaotyu = [0,8,9,17,18,26,27,28,29,30,31,32,33]
        zihai = [27,28,29,30,31,32,33]
        yakuhai = [31,32,33]
        yakuhai_simotya = [31,32,33]
        yakuhai_toimen = [31,32,33]
        yakuhai_kamitya = [31,32,33]
        # 自風,場風を役牌に追加
        if dealer[0][0]==1:
            dealer_num=0
        elif dealer[1][0]==1:
            dealer_num=1
        elif dealer[2][0]==1:
            dealer_num=2
        elif dealer[3][0]==1:
            dealer_num=3

        if dealer_num==0:
            yakuhai.append(27)
            yakuhai_simotya.append(28)
            yakuhai_toimen.append(29)
            yakuhai_kamitya.append(30)
        elif dealer_num==1:
            yakuhai.append(30)
            yakuhai_simotya.append(27)
            yakuhai_toimen.append(28)
            yakuhai_kamitya.append(29)
        elif dealer_num==2:
            yakuhai.append(29)
            yakuhai_simotya.append(30)
            yakuhai_toimen.append(27)
            yakuhai_kamitya.append(28)
        elif dealer_num==3:
            yakuhai.append(28)
            yakuhai_simotya.append(29)
            yakuhai_toimen.append(30)
            yakuhai_kamitya.append(27)
        if round[3][0]==1:
            if not (28 in yakuhai):
                yakuhai.append(28)
            if not (28 in yakuhai_simotya):
                yakuhai_simotya.append(28)
            if not (28 in yakuhai_toimen):
                yakuhai_toimen.append(28)
            if not (28 in yakuhai_kamitya):
                yakuhai_kamitya.append(28)

        else:
            if not (27 in yakuhai):
                yakuhai.append(27)
            if not (27 in yakuhai_simotya):
                yakuhai_simotya.append(27)
            if not (27 in yakuhai_toimen):
                yakuhai_toimen.append(27)
            if not (27 in yakuhai_kamitya):
                yakuhai_kamitya.append(27)
        
        if round[0][0]==0 and dealer_num==0:
            self.my_player_id = 0
        elif round[0][0]==0 and dealer_num==1:
            self.my_player_id = 3
        elif round[0][0]==0 and dealer_num==2:
            self.my_player_id = 2
        elif round[0][0]==0 and dealer_num==3:
            self.my_player_id = 1
        
        for i in range(5):
            if kyotaku[i][0]==1:
                kyotaku_num += 1
            if honba[i][0]==1:
                honba_num += 1
        effective_draw_types = []
        for i in range(34):
            if effective_draw[i]==1:
                effective_draw_types.append(i)
        my_discarded_tiles_types = []
        for i in range(3):
            for j in range(34):
                if discarded_tiles[i][j]==1:
                    if not (j in my_discarded_tiles_types):
                        my_discarded_tiles_types.append(j)
        dora_num_in_hand = 0
        for i in range(13):
            if dora_in_hand[i][0]==1:
                dora_num_in_hand += 1
        dora_total_num += dora_num_in_hand
        
        for i in range(3):
            if ranking[i][0]==1:
                my_rank += 1
        for i in range(3,6):
            if ranking[i][0]==1:
                simotya_rank += 1
        for i in range(6,9):
            if ranking[i][0]==1:
                toimen_rank += 1
        for i in range(9,12):
            if ranking[i][0]==1:
                kamitya_rank += 1
       
        # 山に残っている牌の種類,数をカウント
        self.remaining_tiles = [[4 for _ in range(34)],[1,1,1]]
        for dora in doras:
            opened_dora = 0
            if not dora in [0,9,18,27,31]:
                opened_dora = dora-1
            elif dora==0:
                opened_dora = 8
            elif dora==9:
                opened_dora = 17
            elif dora==18:
                opened_dora = 26
            elif dora==27:
                opened_dora = 31
            elif dora==31:
                opened_dora = 33   
            self.remaining_tiles[0][opened_dora] -= 1

        check_red_dora = [0,0,0]
        if hand[4][0]==1:
            check_red_dora[0]=1
        elif hand[5][0]==1:
            check_red_dora[1]=1
        elif hand[6][0]==1:
            check_red_dora[2]=1
        self.remaining_tiles = [self.remaining_tiles[0], [x-y for (x,y) in zip(self.remaining_tiles[1],check_red_dora)]]
        for i in range(4):
            self.remaining_tiles = [[x-y for (x,y) in zip(self.remaining_tiles[0],hand[i])],self.remaining_tiles[1]]

        self.remaining_tiles_num = 69
        ignored_discards = [0 for _ in range(34)]

        for e in obs.events():
            if e.type() in [EventType.ABORTIVE_DRAW_NORMAL, EventType.ABORTIVE_DRAW_NAGASHI_MANGAN, 
                            EventType.ABORTIVE_DRAW_FOUR_WINDS, EventType.ABORTIVE_DRAW_FOUR_KANS,
                            EventType.ABORTIVE_DRAW_THREE_RONS, EventType.ABORTIVE_DRAW_FOUR_RIICHIS,
                            EventType.RON, EventType.TSUMO]:
                self.when_riichi = [-1,-1,-1]
                self.before_riichi_discards_list = [[0 for _ in range(34)] for _ in range(3)]
                self.ankan_num = 0
                self.ankan_types = []
                is_reset = True

            if e.type() == EventType.RIICHI:
                if self.my_player_id==0:
                    if e.who()==1:
                        riichi[1][0] = 1
                    elif e.who()==2:
                        riichi[2][0] = 1
                    elif e.who()==3:
                        riichi[3][0] = 1
                elif self.my_player_id==1:
                    if e.who()==2:
                        riichi[1][0] = 1
                    elif e.who()==3:
                        riichi[2][0] = 1
                    if e.who()==0:
                        riichi[3][0] = 1
                elif self.my_player_id==2:
                    if e.who()==3:
                        riichi[1][0] = 1
                    elif e.who()==0:
                        riichi[2][0] = 1
                    elif e.who()==1:
                        riichi[3][0] = 1
                elif self.my_player_id==3:
                    if e.who()==0:
                        riichi[1][0] = 1
                    elif e.who()==1:
                        riichi[2][0] = 1
                    elif e.who()==2:
                        riichi[3][0] = 1

            if e.type() == EventType.DISCARD or e.type() == EventType.TSUMOGIRI:
                self.remaining_tiles_num -= 1
                self.remaining_tiles[0][e.tile().type()] -= 1
                ignored_discards[e.tile().type()] += 1
                if e.tile().is_red() and e.tile().type() == 4:
                    self.remaining_tiles[1][0]=0
                elif e.tile().is_red() and e.tile().type() == 13:
                    self.remaining_tiles[1][1]=0
                elif e.tile().is_red() and e.tile().type() == 22:
                    self.remaining_tiles[1][2]=0

            elif e.type() in [
                EventType.CHI,
                EventType.PON,
                EventType.OPEN_KAN]:
                if e.type() != EventType.OPEN_KAN:
                    self.remaining_tiles_num += 1
                for t in e.open().tiles():
                    self.remaining_tiles[0][t.type()] -= 1
                    if t.is_red() and t.type() == 4:
                        self.remaining_tiles[1][0]=0
                    elif t.is_red() and t.type() == 13:
                        self.remaining_tiles[1][1]=0
                    elif t.is_red() and t.type() == 22:
                        self.remaining_tiles[1][2]=0
                self.remaining_tiles[0][e.open().last_tile().type()] += 1
            
            elif e.type() == EventType.CLOSED_KAN:
                self.remaining_tiles_num -= 1
                for t in e.open().tiles():
                    self.remaining_tiles[0][t.type()] -= 1
                    if t.is_red() and t.type() == 4:
                        self.remaining_tiles[1][0]=0
                    elif t.is_red() and t.type() == 13:
                        self.remaining_tiles[1][1]=0
                    elif t.is_red() and t.type() == 22:
                        self.remaining_tiles[1][2]=0
            
            elif e.type() == EventType.ADDED_KAN:
                self.remaining_tiles_num -= 1
                t = e.open().last_tile()
                self.remaining_tiles[0][t.type()] -= 1
                if t.is_red() and t.type() == 4:
                    self.remaining_tiles[1][0]=0
                elif t.is_red() and t.type() == 13:
                    self.remaining_tiles[1][1]=0
                elif t.is_red() and t.type() == 22:
                    self.remaining_tiles[1][2]=0
        
        is_last_round_last_rank = False
        if (round[6][0]==1 and ranking[2][0]==1) or (round[4][0]==1 and tens[self.my_player_id]<=(3000+honba_num*100)) or (tens[self.my_player_id]<=(2000+honba_num*100)) or (my_rank==3 and ((riichi[1][0]==1 and simotya_rank==4) or (riichi[2][0]==1 and toimen_rank==4) or (riichi[3][0]==1 and kamitya_rank==4)) and round[6][0]==1 and ((-diff_ten(tens,3,4))<3900+kyotaku_num*1000+honba_num*300)):
            is_last_round_last_rank = True

        after_riichi_discards_list = [[0 for _ in range(34)] for _ in range(3)]
        # リーチをした順目を記憶
        if not is_reset:
            if riichi[1][0]==1 and self.when_riichi[0]==-1:
                self.when_riichi[0] = self.remaining_tiles_num
                self.before_riichi_discards_list[0] = copy.deepcopy(ignored_discards)
            elif riichi[2][0]==1 and self.when_riichi[1]==-1:
                self.when_riichi[1] = self.remaining_tiles_num
                self.before_riichi_discards_list[1] = copy.deepcopy(ignored_discards)
            elif riichi[3][0]==1 and self.when_riichi[2]==-1:
                self.when_riichi[2] = self.remaining_tiles_num
                self.before_riichi_discards_list[2] = copy.deepcopy(ignored_discards)

            for i in range(3):
                for j in range(34):
                    if riichi[i+1][0]==1:
                        after_riichi_discards_list[i][j] = ignored_discards[j]-self.before_riichi_discards_list[i][j]

        # 行動選択処理
        self.action_mode = ActionModeType.MENZEN
        self.target_yaku = TargetYakuType.NO_TARGET
        dangerous_situation = riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1 or self.remaining_tiles_num<15 # 誰かがリーチ、または流局間際である
        legal_actions = obs.legal_actions()

        # 選択肢が1つであれば,それをする
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 国士無双を目指すかどうかの判断
        kyusyu_actions = [a for a in legal_actions if a.type() == ActionType.ABORTIVE_DRAW_NINE_TERMINALS]
        if len(kyusyu_actions) >= 1:
            # 1位,2位のときは流局させる
            if my_rank>=3:
                self.action_mode = ActionModeType.KOKUSHI
            else:
                count_kyusyu = 0
                for i in yaotyu:
                    if hand[0][i]==1:
                        count_kyusyu += 1
                if count_kyusyu>=11:
                    self.action_mode = ActionModeType.KOKUSHI
                else:
                    return kyusyu_actions[0]
        
        # 四暗刻を狙えるかどうかの判定
        is_possible_to_aim_suanko = False
        is_possible_to_aim_suanko_tanki = False
        toitsu_count = 0
        toitsu_types = []
        anko_count = 0
        anko_types = []
        for i in range(34):
            if hand[1][i]==1 and hand[2][i]==0:
                toitsu_count += 1
                toitsu_types.append(i)
            if hand[2][i]==1:
                anko_count += 1
                anko_types.append(i)
        if self.ankan_num > 0:
            anko_count += self.ankan_num
            for i in self.ankan_types:
                anko_types.append(i)
        if anko_count==3 and toitsu_count==2:
            for i in toitsu_types:
                if self.remaining_tiles[0][i]>=1:
                    is_possible_to_aim_suanko = True
        elif anko_count==4:
            is_possible_to_aim_suanko = True
            is_possible_to_aim_suanko_tanki = True
        
        # アガれるときはアガる
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            return win_actions[0]

        # リーチの処理
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if (not is_last_round_last_rank) and (((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==4) or (riichi[2][0]==1 and toimen_rank==4) or (riichi[3][0]==1 and kamitya_rank==4)) and (-diff_ten(tens,my_rank,4))>(12000+kyotaku_num*1000+honba_num*300) and (-diff_ten(tens,my_rank,2))<(8000+kyotaku_num*1000+honba_num*300))
            or ((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==3) or (riichi[2][0]==1 and toimen_rank==3) or (riichi[3][0]==1 and kamitya_rank==3)) and (-diff_ten(tens,my_rank,3))>(12000+kyotaku_num*1000+honba_num*300)  and (-diff_ten(tens,my_rank,2))<(8000+kyotaku_num*1000+honba_num*300))
            or ((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==2 and (-diff_ten(tens,my_rank,simotya_rank)>(8000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==2 and (-diff_ten(tens,my_rank,toimen_rank)>(8000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==2 and (-diff_ten(tens,my_rank,kamitya_rank)>(8000+kyotaku_num*1000+honba_num*100)))))
            or ((my_rank==2 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==4) or (riichi[2][0]==1 and toimen_rank==4) or (riichi[3][0]==1 and kamitya_rank==4)) and (-diff_ten(tens,my_rank,4))>(12000+kyotaku_num*1000+honba_num*300)  and (-diff_ten(tens,my_rank,3))<(8000+kyotaku_num*1000+honba_num*300))
            or ((my_rank==2 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==3 and (-diff_ten(tens,my_rank,simotya_rank)>(8000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==3 and (-diff_ten(tens,my_rank,toimen_rank)>(8000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==3 and (-diff_ten(tens,my_rank,kamitya_rank)>(8000+kyotaku_num*1000+honba_num*100)))))
            or ((my_rank==2 and round[6][0]==1 and dealer_num==0) and ((riichi[1][0]==1 and simotya_rank==1 and (-diff_ten(tens,my_rank,simotya_rank)>(18000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==1 and (-diff_ten(tens,my_rank,toimen_rank)>(18000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==1 and (-diff_ten(tens,my_rank,kamitya_rank)>(18000+kyotaku_num*1000+honba_num*100)))))
            or ((my_rank==2 and round[6][0]==1 and dealer_num!=0) and ((riichi[1][0]==1 and simotya_rank==1 and (-diff_ten(tens,my_rank,simotya_rank)>(12000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==1 and (-diff_ten(tens,my_rank,toimen_rank)>(12000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==1 and (-diff_ten(tens,my_rank,kamitya_rank)>(12000+kyotaku_num*1000+honba_num*100)))))
            ):
            pass
        elif is_possible_to_aim_suanko_tanki:
            pass
        else:
            if len(riichi_actions) >= 1:
                remaining_agarihai_num = 0
                legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
                effective_discard_types = obs.curr_hand().effective_discard_types()
                effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
                for a in effective_draw_types:
                    remaining_agarihai_num+=self.remaining_tiles[0][a]
                if remaining_agarihai_num==0: # アガり牌が存在しない
                    shanpon_count = 0
                    for a in effective_draw_types:
                        if hand[1][a]==1 and hand[2][a]==0:
                            shanpon_count += 1
                    if shanpon_count >=2:
                        change_wait_discard = [a for a in legal_discards if a.tile().type() in effective_draw_types]
                        is_discarded = True
                        return discard(riichi,discarded_tiles,change_wait_discard,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                    if len(effective_discards)>0:
                        is_discarded = True
                        return discard(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                    else:
                        change_wait_discard_types = []
                        for a in effective_draw_types:
                            for e in check_effective_hai(a):
                                if not e in change_wait_discard_types:
                                    change_wait_discard_types.append(e)
                        change_wait_discards = [a for a in legal_discards if a.tile().type() in change_wait_discard_types]
                        if len(change_wait_discards)>0 and riichi[1][0]==0 and riichi[2][0]==0 and riichi[3][0]==0:
                            is_discarded = True
                            return discard(riichi,discarded_tiles,change_wait_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                        else:
                            is_discarded = True
                            return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                else:
                    is_furiten = False
                    for a in effective_draw_types:
                        if a in my_discarded_tiles_types:
                            is_furiten = True
                    if is_furiten:
                        count_kyusyu = 0
                        for i in yaotyu:
                            if hand[0][i]==1:
                                count_kyusyu += 1
                        if dora_num_in_hand>=2 or count_kyusyu>=12 or is_last_round_last_rank:
                            return riichi_actions[0]
                    else:
                        return riichi_actions[0]

        count_kyusyu = 0
        for i in yaotyu:
            if hand[0][i]==1:
                count_kyusyu += 1
        if count_kyusyu>=9:
            self.action_mode = ActionModeType.KOKUSHI
            for i in yaotyu:
                if self.remaining_tiles[0][i]==0 and hand[0][i]==0: # 牌が売り切れていたら
                    self.action_mode = ActionModeType.MENZEN

        # 副露の数をカウント
        count_furo = 0
        count_furo_list = [0 for _ in range(34)]
        for i in range(34):
            for j in range(4):
                if opend_tiles[j][i]==1:
                    count_furo_list[i] += 1
                    if i in doras:
                        dora_total_num += 1
        if opend_tiles[4][0]==1:
            dora_total_num += 1
        if opend_tiles[5][0]==1:
            dora_total_num += 1
        if opend_tiles[6][0]==1:
            dora_total_num += 1
        for i in count_furo_list:
            count_furo += i
        if self.ankan_num > 0:
            count_furo -= self.ankan_num*4
            for i in self.ankan_types:
                count_furo_list[i] = 0
        if count_furo>0:
            self.action_mode = ActionModeType.FURO # 鳴いている
            for i in yakuhai:
                if count_furo_list[i]>=3:
                    self.action_mode = ActionModeType.FURO_YAKUHAI # 既に役がある鳴き
        count_furo_simotya = 0
        count_dora_simotya = 0
        is_yakuhai_furo_simotya = False
        count_furo_list_simotya = [0 for _ in range(34)]
        for i in range(34):
            for j in range(7,11):
                if opend_tiles[j][i]==1:
                    count_furo_list_simotya[i] += 1
                    if i in doras:
                        count_dora_simotya += 1
        if opend_tiles[11][0]==1:
            count_dora_simotya += 1
        if opend_tiles[12][0]==1:
            count_dora_simotya += 1
        if opend_tiles[13][0]==1:
            count_dora_simotya += 1
        for i in count_furo_list_simotya:
            if i==4:
                count_furo_simotya += 3
            else:
                count_furo_simotya += i
        for i in yakuhai_simotya:
            if count_furo_list_simotya[i]>=3:
                is_yakuhai_furo_simotya = True
            # 2鳴き以上, 役牌鳴き, ドラ3見え以上ならリーチ判定
        if is_yakuhai_furo_simotya and count_dora_simotya>=3 and count_furo_simotya>=6:
            riichi[1][0] = 1
        
        count_furo_toimen = 0
        count_dora_toimen = 0
        is_yakuhai_furo_toimen = False
        count_furo_list_toimen = [0 for _ in range(34)]
        for i in range(34):
            for j in range(14,18):
                if opend_tiles[j][i]==1:
                    count_furo_list_toimen[i] += 1
                    if i in doras:
                        count_dora_toimen += 1
        if opend_tiles[18][0]==1:
            count_dora_toimen += 1
        if opend_tiles[19][0]==1:
            count_dora_toimen += 1
        if opend_tiles[20][0]==1:
            count_dora_toimen += 1
        for i in count_furo_list_toimen:
            if i==4:
                count_furo_toimen += 3
            else:
                count_furo_toimen += i
        for i in yakuhai_toimen:
            if count_furo_list_toimen[i]>=3:
                is_yakuhai_furo_toimen = True
            # 2鳴き以上, 役牌鳴き, ドラ3見え以上ならリーチ判定
        if is_yakuhai_furo_toimen and count_dora_toimen>=3 and count_furo_toimen>=6:
            riichi[2][0] = 1
        
        count_furo_kamitya = 0
        count_dora_kamitya = 0
        is_yakuhai_furo_kamitya = False
        count_furo_list_kamitya = [0 for _ in range(34)]
        for i in range(34):
            for j in range(21,25):
                if opend_tiles[j][i]==1:
                    count_furo_list_kamitya[i] += 1
                    if i in doras:
                        count_dora_kamitya += 1
        if opend_tiles[25][0]==1:
            count_dora_kamitya += 1
        if opend_tiles[26][0]==1:
            count_dora_kamitya += 1
        if opend_tiles[27][0]==1:
            count_dora_kamitya += 1
        for i in count_furo_list_kamitya:
            if i==4:
                count_furo_kamitya += 3
            else:
                count_furo_kamitya += i
        for i in yakuhai_kamitya:
            if count_furo_list_kamitya[i]>=3:
                is_yakuhai_furo_kamitya = True
            # 2鳴き以上, 役牌鳴き, ドラ3見え以上ならリーチ判定
        if is_yakuhai_furo_kamitya and count_dora_kamitya>=3 and count_furo_kamitya>=6:
            riichi[3][0] = 1
        

        # 役牌暗刻持ち、かつドラを2枚以上持っていれば鳴きを視野に入れる
        for i in yakuhai:
            if hand[2][i]==1 and dora_num_in_hand>=2:
                self.action_mode = ActionModeType.FURO_YAKUHAI
        
        # 混一色,清一色を目指すかどうかの判断
        manzu_counter = 0
        furo_manzu_counter = 0
        pinzu_counter = 0
        furo_pinzu_counter = 0
        sozu_counter = 0
        furo_sozu_counter = 0
        zihai_counter = 0
        zihai_toitsu_counter = 0
        zihai_anko_counter = 0
        furo_zihai_counter = 0
        for i in range(34):
            for j in range(4):
                if hand[j][i]==1:
                    if i in manzu:
                        manzu_counter+=1
                    elif i in pinzu:
                        pinzu_counter+=1
                    elif i in sozu:
                        sozu_counter+=1
                    elif i in zihai:
                        if j==1:
                            zihai_toitsu_counter += 1
                        elif j==2:
                            zihai_anko_counter += 1
                            zihai_toitsu_counter -= 1
                        zihai_counter+=1
            if count_furo_list[i]==4:
                count_furo_list[i]=3
            if i in manzu:
                manzu_counter += count_furo_list[i]
                furo_manzu_counter += count_furo_list[i]
            elif i in pinzu:
                pinzu_counter += count_furo_list[i]
                furo_pinzu_counter += count_furo_list[i]
            elif i in sozu:
                sozu_counter += count_furo_list[i]
                furo_sozu_counter += count_furo_list[i]
            elif i in zihai:
                zihai_counter += count_furo_list[i]
                furo_zihai_counter += count_furo_list[i]
        
        remaining_zihai_num = 0
        for i in zihai:
            remaining_zihai_num += self.remaining_tiles[0][i]
        if zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=10 and remaining_zihai_num>13 and furo_manzu_counter==0 and furo_pinzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = TargetYakuType.TSUISO
        elif manzu_counter+furo_manzu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and ((manzu_counter>=pinzu_counter and manzu_counter>=sozu_counter) or furo_manzu_counter>0) and furo_pinzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = TargetYakuType.HONITSU_MANZU
            if manzu_counter+furo_manzu_counter>=10 and furo_zihai_counter==0 and zihai_anko_counter==0:
                self.target_yaku = TargetYakuType.CHINITSU_MANZU
        elif pinzu_counter+furo_pinzu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and ((pinzu_counter>=manzu_counter and pinzu_counter>=sozu_counter) or furo_pinzu_counter>0) and furo_manzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = TargetYakuType.HONITSU_PINZU
            if pinzu_counter+furo_pinzu_counter>=10 and furo_zihai_counter==0 and zihai_anko_counter==0:
                self.target_yaku = TargetYakuType.CHINITSU_PINZU
        elif sozu_counter+furo_sozu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and ((sozu_counter>=manzu_counter and sozu_counter>=pinzu_counter) or furo_sozu_counter>0) and furo_manzu_counter==0 and furo_pinzu_counter==0:
            self.target_yaku = TargetYakuType.HONITSU_SOZU
            if sozu_counter+furo_sozu_counter>=10 and furo_zihai_counter==0 and zihai_anko_counter==0:
                self.target_yaku = TargetYakuType.CHINITSU_SOZU

        
        # 国士無双用の例外処理
        if self.action_mode == ActionModeType.KOKUSHI:
            # ポン,チー,カンの鳴きはパス
            steal_actions = [a for a in legal_actions if a.type() in [ActionType.PON, ActionType.CHI, ActionType.OPEN_KAN]]
            if len(steal_actions) >= 1:
                pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
                return pass_action
            
            legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
            tyutyan_discards = []
            tyutyan_dora_discards = []
            for a in legal_discards:
                if a.tile().type() in tyutyan:
                    tyutyan_discards.append(a)
                    if a.tile().type() in doras or a.tile().is_red():
                        tyutyan_dora_discards.append(a)
            if len(tyutyan_dora_discards)>0:
                tyutyan_discards = tyutyan_dora_discards

            if len(tyutyan_discards)>0:
                is_discarded = True
                return discard(riichi,discarded_tiles,tyutyan_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
            else:
                toitsu_or_anko_yaotyu_list = []
                for i in yaotyu:
                    if hand[1][i] == 1:
                        toitsu_or_anko_yaotyu_list.append(i)
                if len(toitsu_or_anko_yaotyu_list)>=2:
                    yaotyu_discards = [a for a in legal_discards if a.tile().type() in toitsu_or_anko_yaotyu_list]
                    is_discarded = True
                    return discard(riichi,discarded_tiles,yaotyu_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                effective_discard_types = obs.curr_hand().effective_discard_types()
                effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
                if len(effective_discards) > 0:
                    if len(effective_discards) > 1:
                        removed_list = []
                        for a in effective_discards:
                            if a.tile().type() in doras or a.tile().is_red():
                                removed_list.append(a)
                        if len(removed_list)<=0:
                            removed_list = effective_discards
                        is_discarded = True
                        return discard(riichi,discarded_tiles,removed_list,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                    else:
                        is_discarded = True
                        return effective_discards[0]
                is_discarded = True
                return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)

        # ポンの処理
        pon_actions = [a for a in legal_actions if a.type() == ActionType.PON]
        if len(pon_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            if self.target_yaku==TargetYakuType.TSUISO:
                if pon_actions[0].open().last_tile().type() in zihai:
                    return pon_actions[0]
                else:
                    return pass_action

            if self.target_yaku==TargetYakuType.CHINITSU_MANZU:
                if pon_actions[0].open().last_tile().type() in manzu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku==TargetYakuType.HONITSU_MANZU:
                if (pon_actions[0].open().last_tile().type() in manzu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku==TargetYakuType.CHINITSU_PINZU:
                if pon_actions[0].open().last_tile().type() in pinzu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku==TargetYakuType.HONITSU_PINZU:
                if (pon_actions[0].open().last_tile().type() in pinzu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku==TargetYakuType.CHINITSU_SOZU:
                if pon_actions[0].open().last_tile().type() in sozu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku==TargetYakuType.HONITSU_SOZU:
                if (pon_actions[0].open().last_tile().type() in sozu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action

            if pon_actions[0].open().last_tile().type() in effective_draw_types:
                if self.action_mode==ActionModeType.FURO_YAKUHAI or pon_actions[0].open().last_tile().type() in yakuhai:
                    if is_last_round_last_rank and (pon_actions[0].open().last_tile().type() in yakuhai) and (not self.action_mode in [ActionModeType.FURO,ActionModeType.FURO_YAKUHAI]) and (not is_possible_to_rank_up(tens,my_rank,dora_total_num,dealer_num,kyotaku,honba)):
                        return pass_action
                    return pon_actions[0]
            
            # 大三元用の処理
            if (hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)):
                if pon_actions[0].open().last_tile().type()==33:
                    return pon_actions[0]
            elif (hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types)):
                if pon_actions[0].open().last_tile().type()==32:
                    return pon_actions[0]
            elif (hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)) and (hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types)):
                if pon_actions[0].open().last_tile().type()==31:
                    return pon_actions[0]
            elif (((31 in toitsu_types) or hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and ((32 in toitsu_types) or hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types))):
                if pon_actions[0].open().last_tile().type() in [31,32,33]:
                    return pon_actions[0]
            elif (((31 in toitsu_types) or hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and ((33 in toitsu_types) or hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types))):
                if pon_actions[0].open().last_tile().type() in [31,32,33]:
                    return pon_actions[0]
            elif (((32 in toitsu_types) or hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)) and ((33 in toitsu_types) or hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types))):
                if pon_actions[0].open().last_tile().type() in [31,32,33]:
                    return pon_actions[0]

            count_toitsu = 0
            toitsu_feat = []
            for i in range(34):
               if hand[1][i]==1 and hand[2][i]==0:
                count_toitsu += 1
                toitsu_feat.append(i)

            if count_toitsu==1:
                if toitsu_feat[0] in yakuhai:
                    if is_last_round_last_rank and (not self.action_mode in [ActionModeType.FURO,ActionModeType.FURO_YAKUHAI]) and (not is_possible_to_rank_up(tens,my_rank,dora_total_num,dealer_num,kyotaku,honba)):
                        return pass_action
                    return pon_actions[0]
            elif count_toitsu>=2:
                for a in toitsu_feat:
                    if (a in yakuhai and self.remaining_tiles[0][a] != 0) or self.action_mode==ActionModeType.FURO_YAKUHAI:
                        if (a in yakuhai) and (self.remaining_tiles[0][a] != 0) and self.action_mode==ActionModeType.FURO:
                            return pon_actions[0]
                        if is_last_round_last_rank and (a in yakuhai) and (self.remaining_tiles[0][a] != 0) and self.action_mode!=ActionModeType.FURO_YAKUHAI:
                            if is_possible_to_rank_up(tens,my_rank,dora_total_num,dealer_num,kyotaku,honba):
                                return pon_actions[0]
                            else:
                                return pass_action
                        return pon_actions[0]

            return pass_action

        # チーの処理
        chi_actions = [a for a in legal_actions if a.type() == ActionType.CHI]
        if len(chi_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            last_tile = chi_actions[0].open().last_tile()
            if self.target_yaku==TargetYakuType.TSUISO:
                return pass_action
            elif self.target_yaku==TargetYakuType.CHINITSU_MANZU or self.target_yaku==TargetYakuType.HONITSU_MANZU:
                if last_tile.type() in manzu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action
            elif self.target_yaku==TargetYakuType.CHINITSU_PINZU or self.target_yaku==TargetYakuType.HONITSU_PINZU:
                if last_tile.type() in pinzu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action
            elif self.target_yaku==TargetYakuType.CHINITSU_SOZU or self.target_yaku==TargetYakuType.HONITSU_SOZU:
                if last_tile.type() in sozu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action

            if last_tile.type() in effective_draw_types:
                if self.action_mode==ActionModeType.FURO_YAKUHAI:
                    return chi_actions[0]
                else:
                    return pass_action

            if (last_tile.type() in doras) or (last_tile.is_red()):
                doras_of_chi_actions = [0 for _ in range(len(chi_actions))]
                for i in range(len(chi_actions)):
                    a = chi_actions[i]
                    if a.open().tiles_from_hand()[0].type() in doras:
                        doras_of_chi_actions[i] += 1
                        if a.open().tiles_from_hand()[0].is_red():
                            doras_of_chi_actions[i] += 1
                    if a.open().tiles_from_hand()[1].type() in doras:
                        doras_of_chi_actions[i] += 1
                        if a.open().tiles_from_hand()[1].is_red():
                            doras_of_chi_actions[i] += 1
                max_doras_of_chi_actions = max(doras_of_chi_actions)
                max_doras_index_of_chi_actions = doras_of_chi_actions.index(max_doras_of_chi_actions)
                is_last_tile_having = False
                is_anko_tile_having = False
                if hand[0][last_tile.type()] == 1:
                    is_last_tile_having = True
                    if last_tile.type() in doras or last_tile.is_red():
                        is_last_tile_having = False
                if hand[2][chi_actions[max_doras_index_of_chi_actions].open().tiles_from_hand()[0].type()] == 1 or hand[2][chi_actions[max_doras_index_of_chi_actions].open().tiles_from_hand()[1].type()] == 1:
                    is_anko_tile_having = True
                    if last_tile.type() in doras or last_tile.is_red():
                        is_anko_tile_having = False
                
                if self.action_mode==ActionModeType.FURO_YAKUHAI and (not is_last_tile_having) and (not is_anko_tile_having):
                    return chi_actions[max_doras_index_of_chi_actions]
                else:
                    return pass_action
            else:
                return pass_action

        # 明槓の処理
        open_kan_actions = [a for a in legal_actions if a.type() == ActionType.OPEN_KAN]
        if len(open_kan_actions) >= 1:
            if self.action_mode==ActionModeType.FURO_YAKUHAI and (riichi[1][0]==0 and riichi[2][0]==0 and riichi[3][0]==0):
                return open_kan_actions[0]
            else:
                pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
                return pass_action

        # 加槓の処理
        added_kan_actions = [a for a in legal_actions if a.type() == ActionType.ADDED_KAN]
        if len(added_kan_actions) >= 1:
            if shanten[2][0]==1 and (riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1):
                pass
            else:
                return added_kan_actions[0]

        # 暗槓の処理
        closed_kan_actions = [a for a in legal_actions if a.type() == ActionType.CLOSED_KAN]
        if len(closed_kan_actions) >= 1:
            if riichi[0][0]==1:
                self.ankan_num += 1
                self.ankan_types.append(closed_kan_actions[0].open().tiles_from_hand()[0].type())
                return closed_kan_actions[0]
            tile_type_of_closed_kan = closed_kan_actions[0].open().tiles_from_hand()[0].type()
            effective_discard_types = obs.curr_hand().effective_discard_types()
            if tile_type_of_closed_kan in zihai:
                if riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1:
                    pass
                else:
                    self.ankan_num += 1
                    self.ankan_types.append(closed_kan_actions[0].open().tiles_from_hand()[0].type())
                    return closed_kan_actions[0]
            # 順子の一部になっているときにはカンしない
            elif is_piece_of_syuntsu(hand,tile_type_of_closed_kan):
                pass
            elif tile_type_of_closed_kan in effective_discard_types:
                if (riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[2][0]==1:
                    pass
                else:
                    self.ankan_num += 1
                    self.ankan_types.append(closed_kan_actions[0].open().tiles_from_hand()[0].type())
                    return closed_kan_actions[0]
            else:
                if (riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[2][0]==1:
                    pass
                else:
                    self.ankan_num += 1
                    self.ankan_types.append(closed_kan_actions[0].open().tiles_from_hand()[0].type())
                    return closed_kan_actions[0]

        # 打牌の処理
        if not is_discarded:
            adjust_by_dora = 0
            if dora_total_num>=4:
                adjust_by_dora = 1
            legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
            effective_discard_types = obs.curr_hand().effective_discard_types()
            effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
            is_having_anpai_for_simotya = False
            anpai_types_for_simotya = []
            if riichi[1][0]==1:
                for i in range(3,6):
                    for j in range(34):
                        if discarded_tiles[i][j]==1:
                            if j in effective_discard_types:
                                is_having_anpai_for_simotya = True
                                if not j in anpai_types_for_simotya:
                                    anpai_types_for_simotya.append(j)
                for i in range(34):
                    if after_riichi_discards_list[0][i]>0:
                        if i in effective_discard_types:
                            is_having_anpai_for_simotya = True
                            if not i in anpai_types_for_simotya:
                                anpai_types_for_simotya.append(i)
            is_having_anpai_for_toimen = False
            anpai_types_for_toimen = []
            if riichi[2][0]==1:
                for i in range(6,9):
                    for j in range(34):
                        if discarded_tiles[i][j]==1:
                            if j in effective_discard_types:
                                is_having_anpai_for_toimen = True
                                if not j in anpai_types_for_toimen:
                                    anpai_types_for_toimen.append(j)
                for i in range(34):
                    if after_riichi_discards_list[1][i]>0:
                        if i in effective_discard_types:
                            is_having_anpai_for_toimen = True
                            if not i in anpai_types_for_toimen:
                                anpai_types_for_toimen.append(i)
            is_having_anpai_for_kamitya = False
            anpai_types_for_kamitya = []
            if riichi[3][0]==1:
                for i in range(9,12):
                    for j in range(34):
                        if discarded_tiles[i][j]==1:
                            if j in effective_discard_types:
                                is_having_anpai_for_kamitya = True
                                if not j in anpai_types_for_kamitya:
                                    anpai_types_for_kamitya.append(j)
                for i in range(34):
                    if after_riichi_discards_list[2][i]>0:
                        if i in effective_discard_types:
                            is_having_anpai_for_kamitya = True
                            if not i in anpai_types_for_kamitya:
                                anpai_types_for_kamitya.append(i)
            is_having_anpai_for_simotya_and_toimen = False
            is_having_anpai_for_simotya_and_kamitya = False
            is_having_anpai_for_toimen_and_kamitya = False
            is_having_anpai_for_all = False
            if riichi[1][0]==1 and riichi[2][0]==1:
                for s in anpai_types_for_simotya:
                    for t in anpai_types_for_toimen:
                        if s==t:
                            is_having_anpai_for_simotya_and_toimen = True
            if riichi[1][0]==1 and riichi[3][0]==1:
                for s in anpai_types_for_simotya:
                    for k in anpai_types_for_kamitya:
                        if s==k:
                            is_having_anpai_for_simotya_and_kamitya = True
            if riichi[2][0]==1 and riichi[3][0]==1:
                for t in anpai_types_for_toimen:
                    for k in anpai_types_for_kamitya:
                        if t==k:
                            is_having_anpai_for_toimen_and_kamitya = True
            if riichi[1][0]==1 and riichi[2][0]==1 and riichi[3][0]==1:
                for s in anpai_types_for_simotya:
                    for t in anpai_types_for_toimen:
                        for k in anpai_types_for_kamitya:
                            if s==t and t==k:
                                is_having_anpai_for_all = True

            # ベタ降り
            if (not is_last_round_last_rank) and (((riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[2+adjust_by_dora][0]==1 and dealer_num!=0)
                or ((riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[3+adjust_by_dora][0]==1 and dealer_num==0)
                or (((riichi[1][0]==1 and riichi[2][0]==1) or (riichi[1][0]==1 and riichi[3][0]==1) or (riichi[2][0]==1 and riichi[3][0]==1)) and shanten[1+adjust_by_dora][0]==1)
                or ((riichi[1][0]==1 and riichi[2][0]==1 and riichi[3][0]==1) and shanten[0][0]==1)
                or (((riichi[1][0]==1 and dealer_num==1) or (riichi[2][0]==1 and dealer_num==2) or (riichi[3][0]==1 and dealer_num==3)) and shanten[1+adjust_by_dora][0]==1)
                or (round[5][0]==1 and (riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[0][0]==1 and (my_rank==1 and (-diff_ten(tens,my_rank,2))>18000))
                or ((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==4) or (riichi[2][0]==1 and toimen_rank==4) or (riichi[3][0]==1 and kamitya_rank==4)) and (-diff_ten(tens,my_rank,4))>(12000+kyotaku_num*1000+honba_num*300) and shanten[0][0]==1 and (-diff_ten(tens,my_rank,2))<(8000+kyotaku_num*1000+honba_num*300))
                or ((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==3) or (riichi[2][0]==1 and toimen_rank==3) or (riichi[3][0]==1 and kamitya_rank==3)) and (-diff_ten(tens,my_rank,3))>(12000+kyotaku_num*1000+honba_num*300) and shanten[0][0]==1 and (-diff_ten(tens,my_rank,2))<(8000+kyotaku_num*1000+honba_num*300))
                or ((my_rank==1 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==2 and (-diff_ten(tens,my_rank,simotya_rank)>(8000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==2 and (-diff_ten(tens,my_rank,toimen_rank)>(8000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==2 and (-diff_ten(tens,my_rank,kamitya_rank)>(8000+kyotaku_num*1000+honba_num*100)))) and shanten[0][0]==1)
                or ((my_rank==2 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==4) or (riichi[2][0]==1 and toimen_rank==4) or (riichi[3][0]==1 and kamitya_rank==4)) and (-diff_ten(tens,my_rank,4))>(12000+kyotaku_num*1000+honba_num*300) and shanten[0][0]==1 and (-diff_ten(tens,my_rank,3))<(8000+kyotaku_num*1000+honba_num*300))
                or ((my_rank==2 and round[6][0]==1) and ((riichi[1][0]==1 and simotya_rank==3 and (-diff_ten(tens,my_rank,simotya_rank)>(8000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==3 and (-diff_ten(tens,my_rank,toimen_rank)>(8000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==3 and (-diff_ten(tens,my_rank,kamitya_rank)>(8000+kyotaku_num*1000+honba_num*100)))) and shanten[0][0]==1)
                or ((my_rank==2 and round[6][0]==1 and dealer_num==0) and ((riichi[1][0]==1 and simotya_rank==1 and (-diff_ten(tens,my_rank,simotya_rank)>(18000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==1 and (-diff_ten(tens,my_rank,toimen_rank)>(18000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==1 and (-diff_ten(tens,my_rank,kamitya_rank)>(18000+kyotaku_num*1000+honba_num*100)))) and shanten[0][0]==1)
                or ((my_rank==2 and round[6][0]==1 and dealer_num!=0) and ((riichi[1][0]==1 and simotya_rank==1 and (-diff_ten(tens,my_rank,simotya_rank)>(12000+kyotaku_num*1000+honba_num*400))) or (riichi[2][0]==1 and toimen_rank==1 and (-diff_ten(tens,my_rank,toimen_rank)>(12000+kyotaku_num*1000+honba_num*100))) or (riichi[3][0]==1 and kamitya_rank==1 and (-diff_ten(tens,my_rank,kamitya_rank)>(12000+kyotaku_num*1000+honba_num*100)))) and shanten[0][0]==1)
                ):
                is_discarded = True
                if len(effective_discards)>0:
                    if riichi[1][0]==1 and is_having_anpai_for_simotya and riichi[2][0]==0 and riichi[3][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[2][0]==1 and is_having_anpai_for_toimen and riichi[1][0]==0 and riichi[3][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[3][0]==1 and is_having_anpai_for_kamitya and riichi[1][0]==0 and riichi[2][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[1][0]==1 and riichi[2][0]==1 and is_having_anpai_for_simotya_and_toimen and riichi[3][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[1][0]==1 and riichi[3][0]==1 and is_having_anpai_for_simotya_and_kamitya and riichi[2][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[2][0]==1 and riichi[3][0]==1 and is_having_anpai_for_toimen_and_kamitya and riichi[1][0]==0:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                    elif riichi[1][0]==1 and riichi[2][0]==1 and riichi[3][0]==1 and is_having_anpai_for_all:
                        return discard_in_riichi(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
                else:
                    return discard_in_riichi(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi)
            for i in yakuhai: # 既に鳴いているときは対子役牌を捨てない
                for a in effective_discards:
                    if a.tile().type()==i:
                        if hand[1][i]==1 and hand[2][i]==0 and self.action_mode==ActionModeType.FURO and (not self.target_yaku in [TargetYakuType.CHINITSU_MANZU,TargetYakuType.CHINITSU_PINZU,TargetYakuType.CHINITSU_SOZU]):
                            effective_discards.remove(a)
            for i in zihai: # 対子で持っているが売り切れている字牌は捨てる
                if (hand[1][i]==1 and hand[2][i]==0) and self.remaining_tiles[0][i]==0:
                    effective_zihai_discards = [a for a in legal_discards if a.tile().type() == i]
                    for a in effective_zihai_discards:
                        effective_discards.append(a)
            for i in yakuhai: # 暗刻で持っている役牌は捨てない
                if (hand[2][i]==1):
                    yakuhai_anko_discard = [a for a in legal_discards if a.tile().type() == i]
                    for a in yakuhai_anko_discard:
                        if a in effective_discards:
                            effective_discards.remove(a)
            # 大三元を目指すための条件
            if (hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)):
                if hand[0][33]==1 and self.remaining_tiles[0][33]!=0:
                    if (33 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 33]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)

            elif (hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types)):
                if hand[0][32]==1 and self.remaining_tiles[0][32]!=0:
                    if (32 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 32]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
            elif (hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)) and (hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types)):
                if hand[0][31]==1 and self.remaining_tiles[0][31]!=0:
                    if (31 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 31]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
            elif ((((31 in toitsu_types) and (self.remaining_tiles[0][31]>0)) or hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (((32 in toitsu_types) and (self.remaining_tiles[0][32]>0)) or hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types))):
                if hand[0][33]==1 and self.remaining_tiles[0][33]!=0:
                    if (31 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 31]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (32 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 32]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (33 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 33]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
            elif ((((31 in toitsu_types) and (self.remaining_tiles[0][31]>0)) or hand[2][31]==1 or count_furo_list[31]>0 or (31 in self.ankan_types)) and (((33 in toitsu_types) and (self.remaining_tiles[0][33]>0)) or hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types))):
                if hand[0][32]==1 and self.remaining_tiles[0][32]!=0:
                    if (31 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 31]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (32 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 32]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (33 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 33]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
            elif ((((32 in toitsu_types) and (self.remaining_tiles[0][32]>0)) or hand[2][32]==1 or count_furo_list[32]>0 or (32 in self.ankan_types)) and (((33 in toitsu_types) and (self.remaining_tiles[0][33]>0)) or hand[2][33]==1 or count_furo_list[33]>0 or (33 in self.ankan_types))):
                if hand[0][31]==1 and self.remaining_tiles[0][31]!=0:
                    if (31 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 31]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (32 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 32]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
                    if (33 in effective_discard_types):
                        yakuhai_discard = [a for a in legal_discards if a.tile().type() == 33]
                        for a in yakuhai_discard:
                            if a in effective_discards:
                                effective_discards.remove(a)
            
            if len(effective_discards) <= 0:
                effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]

            if self.target_yaku==TargetYakuType.CHINITSU_MANZU:
                tinitsu_discards = [a for a in legal_discards if (a.tile().type() in pinzu) or (a.tile().type() in sozu) or (a.tile().type() in zihai)]
                if len(tinitsu_discards)>0:
                    effective_discards = tinitsu_discards
            elif self.target_yaku==TargetYakuType.CHINITSU_PINZU:
                tinitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in sozu) or (a.tile().type() in zihai)]
                if len(tinitsu_discards)>0:
                    effective_discards = tinitsu_discards
            elif self.target_yaku==TargetYakuType.CHINITSU_SOZU:
                tinitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in pinzu) or (a.tile().type() in zihai)]
                if len(tinitsu_discards)>0:
                    effective_discards = tinitsu_discards
            elif self.target_yaku==TargetYakuType.HONITSU_MANZU:
                honitsu_discards = [a for a in legal_discards if (a.tile().type() in pinzu) or (a.tile().type() in sozu)]
                if len(honitsu_discards)>0:
                    effective_discards = honitsu_discards
            elif self.target_yaku==TargetYakuType.HONITSU_PINZU:
                honitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in sozu)]
                if len(honitsu_discards)>0:
                    effective_discards = honitsu_discards
            elif self.target_yaku==TargetYakuType.HONITSU_SOZU:
                honitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in pinzu)]
                if len(honitsu_discards)>0:
                    effective_discards = honitsu_discards
            
            if len(effective_discards) > 0:
                if len(effective_discards) > 1:
                    if is_possible_to_aim_suanko:
                        effective_discards_for_suanko = [a for a in legal_discards if a.tile().type() in effective_discard_types]
                        if anko_count==3 and toitsu_count==2:
                            for a in effective_discards_for_suanko:
                                if (a.tile().type() in anko_types) or (a.tile().type() in toitsu_types):
                                    if a in effective_discards_for_suanko:
                                        effective_discards_for_suanko.remove(a)
                        if anko_count==4:
                            for a in effective_discards_for_suanko:
                                if (a.tile().type() in anko_types):
                                    if a in effective_discards_for_suanko:
                                        effective_discards_for_suanko.remove(a)
                        if len(effective_discards_for_suanko)>0:
                            return discard(riichi,discarded_tiles,effective_discards_for_suanko,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                    elif len(effective_discards)==2 and shanten[1][0]==0:
                        if toitsu_count>=6: # 七対子のリーチ
                            a = effective_discards[0]
                            b = effective_discards[1]
                            count_a_suzi_in_my_discarded_tiles = 0
                            count_b_suzi_in_my_discarded_tiles = 0
                            for x in check_suzi(a.tile().type()):
                                if x in my_discarded_tiles_types:
                                    count_a_suzi_in_my_discarded_tiles += 1
                            for y in check_suzi(b.tile().type()):
                                if y in my_discarded_tiles_types:
                                    count_b_suzi_in_my_discarded_tiles += 1
                            if (a.tile().type() in doras or a.tile().is_red()) and self.remaining_tiles[0][a.tile().type()]!=0 and ((not (b.tile().type() in doras)) and (not b.tile().is_red())):
                                return b
                            elif (b.tile().type() in doras or b.tile().is_red()) and self.remaining_tiles[0][b.tile().type()]!=0 and ((not (a.tile().type() in doras)) and (not a.tile().is_red())):
                                return a
                            elif (a.tile().type() in doras or a.tile().is_red()) and (b.tile().type() in doras or b.tile().is_red()):
                                if (a.tile().type() in doras and a.tile().is_red()) and self.remaining_tiles[0][a.tile().type()]!=0 and (((b.tile().type() in doras) and (not b.tile().is_red())) or (b.tile().is_red() and (not (b.tile().type() in doras)))):
                                    return b
                                elif (b.tile().type() in doras and b.tile().is_red()) and self.remaining_tiles[0][b.tile().type()]!=0 and (((a.tile().type() in doras) and (not a.tile().is_red())) or (a.tile().is_red() and (not (a.tile().type() in doras)))):
                                    return a
                                else:
                                    if a.tile().type() in doras and self.remaining_tiles[0][b.tile().type()]!=0 and b.tile().is_red():
                                        return a
                                    elif b.tile().type() in doras and self.remaining_tiles[0][a.tile().type()]!=0 and a.tile().is_red():
                                        return b
                                    elif self.remaining_tiles[0][a.tile().type()]>self.remaining_tiles[0][b.tile().type()]:
                                        return b
                                    elif self.remaining_tiles[0][a.tile().type()]<=self.remaining_tiles[0][b.tile().type()]:
                                        return a
                            elif self.remaining_tiles[0][a.tile().type()]>=1 and (not (a.tile().type() in my_discarded_tiles_types)) and (count_a_suzi_in_my_discarded_tiles>=count_b_suzi_in_my_discarded_tiles):
                                return b
                            elif self.remaining_tiles[0][b.tile().type()]>=1 and (not (b.tile().type() in my_discarded_tiles_types)) and (count_a_suzi_in_my_discarded_tiles<count_b_suzi_in_my_discarded_tiles):
                                return a
                            elif self.remaining_tiles[0][a.tile().type()]>self.remaining_tiles[0][b.tile().type()]:
                                return b
                            elif self.remaining_tiles[0][a.tile().type()]<=self.remaining_tiles[0][a.tile().type()]:
                                return a
                    effective_discards_removed_doras = [a for a in legal_discards if a.tile().type() in effective_discard_types]
                    for a in effective_discards_removed_doras:
                        if a.tile().type() in doras or a.tile().is_red(): # ドラは捨てない（他の選択肢がある場合）
                            if (a.tile().type() in zihai) and ((hand[0][a.tile().type()]==1 and hand[1][a.tile().type()]==0 and self.remaining_tiles[0][a.tile().type()]>=2) or (hand[1][a.tile().type()]==1 and hand[2][a.tile().type()]==0 and self.remaining_tiles[0][a.tile().type()]>=1)):
                                effective_discards_removed_doras.remove(a)
                            elif a.tile().type() in zihai:
                                pass
                            else:
                                effective_discards_removed_doras.remove(a)
                    if len(effective_discards_removed_doras)<=0:
                        return discard(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                    else:
                        return discard(riichi,discarded_tiles,effective_discards_removed_doras,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
                return discard(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
            # 効果的な打牌がない
            return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,after_riichi_discards_list,self.when_riichi,dangerous_situation,is_last_round_last_rank)
        
class MenzenAgent(CustomAgentBase):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # if it can apply chi/pon/open-kan, choose randomly
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            return pass_action

        # if it can apply closed-kan/added-kan, choose randomly
        kan_actions = [
            a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN, ActionType.ADDED_KAN]
        ]
        if len(kan_actions) >= 1:
            return random.choice(kan_actions)

        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        ]
        effective_discard_types = observation.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]
        if len(effective_discards) > 0:
            return random.choice(effective_discards)

        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)

def save_log(obs_dict, env, logs):
    logdir = "logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    now = datetime.now().strftime('%Y%m%d%H%M%S%f')

    os.mkdir(os.path.join(logdir, now))
    for player_id, obs in obs_dict.items():
        with open(os.path.join(logdir, now, f"{player_id}.json"), "w") as f:
            json.dump(json.loads(obs.to_json()), f)
        with open(os.path.join(logdir, now, f"tenho.log"), "w") as f:
            f.write(logs.get_url())
    env.state().save_svg(os.path.join(logdir, now, "finish.svg"))
    with open(os.path.join(logdir, now, f"env.json"), "w") as f:
        f.write(env.state().to_json())


if __name__ == "__main__":
    """引数
    -n, --number (int): 何回対局するか
    -l --log (flag): このオプションをつけると対局結果を保存する
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=32,
                        help="number of game iteration")
    parser.add_argument("-l", "--log", action="store_true",
                        help="whether log will be stored")
    args = parser.parse_args()

    logging = args.log
    n_games = args.number

    player_names_to_idx ={
        "player_0": 0,
        "player_1": 1,
        "player_2": 2,
        "player_3": 3,
    }

    agents = [                 
        MyAgent(),  # 自作Agent
        MenzenAgent(),
        MenzenAgent(),
        MenzenAgent(),

        # mjx.agents.ShantenAgent(),  # mjxに実装されているAgent 
        # mjx.agents.ShantenAgent(),  # mjxに実装されているAgent 
        # mjx.agents.ShantenAgent(),  # mjxに実装されているAgent 
        ]

    # 卓の初期化
    env_ = mjx.MjxEnv()
    obs_dict = env_.reset()

    my_id = "player_0"
    result_rank = ""

    logs = convert_log.ConvertLog()
    for _ in range(n_games):
        env_ = mjx.MjxEnv()
        obs_dict = env_.reset()
        first_counter = 0
        while not env_.done():
            actions = {}
            for player_id, obs in obs_dict.items():
                actions[player_id] = agents[player_names_to_idx[player_id]].act(obs)
            obs_dict = env_.step(actions)
            if len(obs_dict.keys())==4:
                logs.add_log(obs_dict)
        returns = env_.rewards()
        if env_.rewards()[my_id]==90:
            result_rank = result_rank+"1"
        elif env_.rewards()[my_id]==45:
            result_rank = result_rank+"2"
        elif env_.rewards()[my_id]==0:
            result_rank = result_rank+"3"
        elif env_.rewards()[my_id]==-135:
            result_rank = result_rank+"4"
        else:
            result_rank = result_rank+"?"

        if logging:
            save_log(obs_dict, env_, logs)

    sum_rank = 0
    for i in result_rank:
        sum_rank += int(i)

    print("rank: "+result_rank)
    print("Ave.: "+str(round(sum_rank/n_games,2)))
    print("1st: "+str(round(int(result_rank.count('1'))*100/n_games,2))+"%")
    print("2nd: "+str(round(int(result_rank.count('2'))*100/n_games,2))+"%")
    print("3rd: "+str(round(int(result_rank.count('3'))*100/n_games,2))+"%")
    print("4th: "+str(round(int(result_rank.count('4'))*100/n_games,2))+"%")
    print("game has ended")
