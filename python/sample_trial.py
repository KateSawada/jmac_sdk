"""対局を指定した回数行うスクリプト．対局結果の出力も可能．
"""

import argparse
import os
from datetime import datetime
import json
import random

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

from server import convert_log
from client.agent import CustomAgentBase


# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．


class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.remaining_tiles = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[1,1,1]]
        self.action_mode = "menzen"
        self.target_yaku = ""
        self.remaining_tiles_num = 70
        self.when_riichi = [-1,-1,-1]
    
    
    def act(self, obs: mjx.Observation) -> mjx.Action:                    
        hand = obs.MjxLargeV0().current_hand(obs)
        riichi = obs.MjxLargeV0().under_riichis(obs)
        discarded_tiles = obs.MjxLargeV0().discarded_tiles(obs)
        opend_tiles = obs.MjxLargeV0().opened_tiles(obs)
        dealer = obs.MjxLargeV0().dealer(obs)
        ranking = obs.MjxLargeV0().rankings(obs)
        effective_draw = obs.MjxLargeV0().effective_draws(obs)[0]
        shanten = obs.MjxLargeV0().shanten(obs)
        round = obs.MjxLargeV0().round(obs)
        doras = obs.doras()
        dora_in_hand = obs.MjxLargeV0().dora_num_in_hand(obs)
        dealer_num = -1

        manzu = [0,1,2,3,4,5,6,7,8]
        pinzu = [9,10,11,12,13,14,15,16,17]
        sozu = [18,19,20,21,22,23,24,25,26]
        tyutyan = [1,2,3,4,5,6,7,10,11,12,13,14,15,16,19,20,21,22,23,24,25]
        yaotyu = [0,8,9,17,18,26,27,28,29,30,31,32,33]
        zihai = [27,28,29,30,31,32,33]
        yakuhai = [31,32,33]
        # 自風,場風を役牌に追加
        for a in dealer:
            if a[0]==1:
                dealer_num=0
            elif a[1]==1:
                dealer_num=1
            elif a[2]==1:
                dealer_num=2
            elif a[3]==1:
                dealer_num=3
        if dealer_num==0:
            yakuhai.append(27)
        elif dealer_num==1:
            yakuhai.append(30)
        elif dealer_num==2:
            yakuhai.append(29)
        elif dealer_num==3:
            yakuhai.append(28)
        if round[3][0]==1:
            if not 28 in yakuhai:
                yakuhai.append(28)
        else:
            if not 27 in yakuhai:
                yakuhai.append(27)

        effective_draw_types = []
        for i in range(34):
            if effective_draw[i]==1:
                effective_draw_types.append(i)
        my_discarded_tiles_types = []
        for i in range(3):
            for j in range(34):
                if discarded_tiles[i][j]==1:
                    if j not in my_discarded_tiles_types:
                        my_discarded_tiles_types.append(j)
        dora_num_in_hand = 0
        for i in range(13):
            if dora_in_hand[i][0]==1:
                dora_num_in_hand += 1
        
        # 山に残っている牌の種類,数をカウント
        self.remaining_tiles = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[1,1,1]]
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

        self.remaining_tiles_num = 70

        for e in obs.events():
                if e.type() == EventType.DISCARD or e.type() == EventType.TSUMOGIRI:
                    self.remaining_tiles_num -= 1
                    self.remaining_tiles[0][e.tile().type()] -= 1
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
                
                elif e.type() in [EventType.CLOSED_KAN]:
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
                
                if self.remaining_tiles_num==70:
                    self.when_riichi = [-1,-1,-1]

        print(self.when_riichi)
        # リーチをした順目を記憶
        if riichi[1][0]==1 and self.when_riichi[0]==-1:
            self.when_riichi[0] = self.remaining_tiles_num
        elif riichi[2][0]==1 and self.when_riichi[1]==-1:
            self.when_riichi[1] = self.remaining_tiles_num
        elif riichi[3][0]==1 and self.when_riichi[2]==-1:
            self.when_riichi[2] = self.remaining_tiles_num

        # 行動選択処理
        self.action_mode = "menzen"
        self.target_yaku = ""
        dangerous_situation = riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1 or self.remaining_tiles_num<15 # 誰かがリーチ、または流局間際である
        legal_actions = obs.legal_actions()

        # 選択肢が1つであれば,それをする
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 国士無双を目指すかどうかの判断
        kyusyu_actions = [a for a in legal_actions if a.type() == ActionType.ABORTIVE_DRAW_NINE_TERMINALS]
        if len(kyusyu_actions) >= 1:
            # 1位,2位のときは流局させる
            if ranking[1][0]==1:
                self.action_mode = "kyusyu"
            else:
                count_kyusyu = 0
                for i in yaotyu:
                    if hand[0][i]==1:
                        count_kyusyu += 1
                if count_kyusyu>=10:
                    self.action_mode = "kyusyu"
                else:
                    return kyusyu_actions[0]
        
        # アガれるときはアガる
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            return win_actions[0]

        # リーチできるときはリーチする
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) == 1:
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
                    return discard(riichi,discarded_tiles,change_wait_discard,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
                if len(effective_discards)>0:
                    return discard(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
                else:
                    return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
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
                    if dora_num_in_hand>=2 or count_kyusyu>=12:
                        return riichi_actions[0]
                else:
                    return riichi_actions[0]
        elif len(riichi_actions) > 1:
            list_of_effective_draw = []
            legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
            effective_discard_types = obs.curr_hand().effective_discard_types()
            effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
            if len(effective_discards)>0:
                for a in effective_discards:
                    discard_num = a.tile().type()
                    for i in range(4):
                        if obs.MjxLargeV0().current_hand(obs)[3-i][discard_num]==1:
                            obs.MjxLargeV0().current_hand(obs)[3-i][discard_num]=0
                            ed = obs.MjxLargeV0().effective_draws(obs)[0]
                            list_of_effective_draw.append(ed)
                            obs.MjxLargeV0().current_hand(obs)[3-i][discard_num]=1
                            break
                number_of_effective_draw = [0 for _ in range(len(effective_discards))]
                for i in list_of_effective_draw:
                    for j in i:
                        number_of_effective_draw[list_of_effective_draw.index(i)] += j
                most_effective_riichi_action = riichi_actions[0]
                max_number_of_effective_draw = -1
                for i in number_of_effective_draw:
                    if i > max_number_of_effective_draw:
                        max_number_of_effective_draw = i
                        most_effective_riichi_action = riichi_actions[number_of_effective_draw.index(i)]
                return most_effective_riichi_action
            else:
                return riichi_actions[0]

        count_kyusyu = 0
        for i in yaotyu:
            if hand[0][i]==1:
                count_kyusyu += 1
        if count_kyusyu>=9:
            self.action_mode = "kyusyu"
            for i in yaotyu:
                if self.remaining_tiles[0][i]==0 and hand[0][i]==0: # 牌が売り切れていたら
                    self.action_mode = "menzen"

        # 副露の数をカウント
        count_furo = 0
        count_furo_list = [0 for _ in range(34)]
        for i in range(34):
            for j in range(4):
                if opend_tiles[j][i]==1:
                    count_furo_list[i]+=1
        for i in count_furo_list:
            count_furo += i
        if count_furo>0:
            self.action_mode = "furo" # 鳴いている
            for i in yakuhai:
                if count_furo_list[i]>=3:
                    self.action_mode = "yakuhai_furo" # 既に役がある鳴き
        
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
        if zihai_toitsu_counter*2+zihai_anko_counter*3>=9 and remaining_zihai_num>13 and furo_manzu_counter==0 and furo_pinzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = "tsuiso"
        elif manzu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and manzu_counter>=pinzu_counter and manzu_counter>=sozu_counter and furo_pinzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = "some_m"
            if manzu_counter>=11 and furo_zihai_counter==0:
                self.target_yaku = "tin_m"
        elif pinzu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and pinzu_counter>=manzu_counter and pinzu_counter>=sozu_counter and furo_manzu_counter==0 and furo_sozu_counter==0:
            self.target_yaku = "some_p"
            if pinzu_counter>=11 and furo_zihai_counter==0:
                self.target_yaku = "tin_p"
        elif sozu_counter+zihai_toitsu_counter*2+zihai_anko_counter*3+furo_zihai_counter>=11 and sozu_counter>=manzu_counter and sozu_counter>=pinzu_counter and furo_manzu_counter==0 and furo_pinzu_counter==0:
            self.target_yaku = "some_s"
            if sozu_counter>=11 and furo_zihai_counter==0:
                self.target_yaku = "tin_s"

        
        # 国士無双用の例外処理
        if self.action_mode == "kyusyu":
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
                return discard(riichi,discarded_tiles,tyutyan_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
            else:
                toitsu_or_anko_yaotyu_list = []
                for i in yaotyu:
                    if hand[1][i] == 1:
                        toitsu_or_anko_yaotyu_list.append(i)
                if len(toitsu_or_anko_yaotyu_list)>=2:
                    yaotyu_discards = [a for a in legal_discards if a.tile().type() in toitsu_or_anko_yaotyu_list]
                    return discard(riichi,discarded_tiles,yaotyu_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
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
                        return discard(riichi,discarded_tiles,removed_list,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
                    else:
                        return effective_discards[0]

                return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)

        # ポンの処理
        pon_actions = [a for a in legal_actions if a.type() == ActionType.PON]
        if len(pon_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            if self.target_yaku=="tsuiso":
                if pon_actions[0].open().last_tile().type() in zihai:
                    return pon_actions[0]
                else:
                    return pass_action

            if self.target_yaku=="tin_m":
                if pon_actions[0].open().last_tile().type() in manzu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku=="some_m":
                if (pon_actions[0].open().last_tile().type() in manzu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku=="tin_p":
                if pon_actions[0].open().last_tile().type() in pinzu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku=="some_p":
                if (pon_actions[0].open().last_tile().type() in pinzu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku=="tin_s":
                if pon_actions[0].open().last_tile().type() in sozu and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action
            elif self.target_yaku=="some_s":
                if (pon_actions[0].open().last_tile().type() in sozu or pon_actions[0].open().last_tile().type() in zihai) and pon_actions[0].open().last_tile().type() in effective_draw_types:
                    return pon_actions[0]
                else:
                    return pass_action

            if pon_actions[0].open().last_tile().type() in effective_draw_types:
                if self.action_mode=="yakuhai_furo":
                    return pon_actions[0]

            count_toitsu = 0
            toitsu_feat = []
            for i in range(34):
               if hand[1][i]==1 and hand[2][i]==0:
                count_toitsu += 1
                toitsu_feat.append(i)

            if count_toitsu==1:
                if toitsu_feat[0] in yakuhai:
                    return pon_actions[0]
            elif count_toitsu>=2:
                for a in toitsu_feat:
                    if (a in yakuhai and self.remaining_tiles[0][a] != 0) or self.action_mode=="yakuhai_furo":
                       return pon_actions[0]

            return pass_action

        # チーの処理
        chi_actions = [a for a in legal_actions if a.type() == ActionType.CHI]
        if len(chi_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            last_tile = chi_actions[0].open().last_tile()
            if self.target_yaku=="tsuiso":
                return pass_action
            elif self.target_yaku=="tin_m" or self.target_yaku=="some_m":
                if last_tile.type() in manzu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action
            elif self.target_yaku=="tin_p" or self.target_yaku=="some_p":
                if last_tile.type() in pinzu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action
            elif self.target_yaku=="tin_s" or self.target_yaku=="some_s":
                if last_tile.type() in sozu and last_tile.type() in effective_draw_types:
                    return chi_actions[0]
                else:
                    pass_action

            if last_tile.type() in effective_draw_types:
                if self.action_mode=="yakuhai_furo":
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
                
                if self.action_mode=="yakuhai_furo" and (not is_last_tile_having) and (not is_anko_tile_having):
                    return chi_actions[max_doras_index_of_chi_actions]
                else:
                    return pass_action
            else:
                return pass_action

        # 明槓の処理
        open_kan_actions = [a for a in legal_actions if a.type() in [ActionType.OPEN_KAN]]
        if len(open_kan_actions) >= 1:
            if self.action_mode=="yakuhai_furo":
                return open_kan_actions[0]
            else:
                pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
                return pass_action

        # 加槓の処理
        added_kan_actions = [a for a in legal_actions if a.type() in [ActionType.ADDED_KAN]]
        if len(added_kan_actions) >= 1:
            return added_kan_actions[0]

        # 暗槓の処理
        closed_kan_actions = [a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN]]
        if len(closed_kan_actions) >= 1:
            if riichi[0][0]==1:
                return closed_kan_actions[0]
            tile_type_of_closed_kan = closed_kan_actions[0].open().tiles_from_hand()[0].type()
            effective_discard_types = obs.curr_hand().effective_discard_types()
            if tile_type_of_closed_kan in zihai:
                return closed_kan_actions[0] # 字牌は無条件で暗槓
            # 順子の一部になっているときにはカンしない
            elif is_piece_of_syuntsu(hand,tile_type_of_closed_kan):
                pass
            elif tile_type_of_closed_kan in effective_discard_types:
                return closed_kan_actions[0]
            else:
                return closed_kan_actions[0]

        # 打牌の処理
        legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
        if ((riichi[1][0]==1 or riichi[2][0]==1 or riichi[3][0]==1) and shanten[3][0]==1) or (((riichi[1][0]==1 and dealer_num==1) or (riichi[2][0]==1 and dealer_num==2) or (riichi[3][0]==1 and dealer_num==3)) and shanten[2][0]==1):
            return discard_in_riichi(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi)
        effective_discard_types = obs.curr_hand().effective_discard_types()
        effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
        for i in yakuhai:
            for a in effective_discards:
                if a.tile().type()==i:
                    if hand[1][i]==1 and hand[2][i]==0 and self.action_mode=="furo" and (not self.target_yaku in ["tin_m","tin_p","tin_s"]):
                        effective_discards.remove(a)
        for i in zihai:
            if effective_draw[i]==1 and self.remaining_tiles[0][i]==0:
                effective_zihai_discards = [a for a in legal_discards if a.tile().type() == i]
                for a in effective_zihai_discards:
                    effective_discards.append(a)
        if self.target_yaku=="tin_m":
            tinitsu_discards = [a for a in legal_discards if (a.tile().type() in pinzu) or (a.tile().type() in sozu) or (a.tile().type() in zihai)]
            if len(tinitsu_discards)>0:
                effective_discards = tinitsu_discards
        elif self.target_yaku=="tin_p":
            tinitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in sozu) or (a.tile().type() in zihai)]
            if len(tinitsu_discards)>0:
                effective_discards = tinitsu_discards
        elif self.target_yaku=="tin_s":
            tinitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in pinzu) or (a.tile().type() in zihai)]
            if len(tinitsu_discards)>0:
                effective_discards = tinitsu_discards
        elif self.target_yaku=="some_m":
            honitsu_discards = [a for a in legal_discards if (a.tile().type() in pinzu) or (a.tile().type() in sozu)]
            if len(honitsu_discards)>0:
                effective_discards = honitsu_discards
        elif self.target_yaku=="some_p":
            honitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in sozu)]
            if len(honitsu_discards)>0:
                effective_discards = honitsu_discards
        elif self.target_yaku=="some_s":
            honitsu_discards = [a for a in legal_discards if (a.tile().type() in manzu) or (a.tile().type() in pinzu)]
            if len(honitsu_discards)>0:
                effective_discards = honitsu_discards
        
        if len(effective_discards) > 0:
            if len(effective_discards) > 1:
                for a in effective_discards:
                    if a.tile().type() in doras or a.tile().is_red():
                        effective_discards.remove(a)
                if len(effective_discards)<=0:
                    return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
            return discard(riichi,discarded_tiles,effective_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
        # 効果的な打牌がない
        return discard(riichi,discarded_tiles,legal_discards,dealer_num,doras,self.remaining_tiles,self.when_riichi,dangerous_situation)
    
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
        MyAgent(),                  # 自作Agent
        mjx.agents.ShantenAgent(),  # mjxに実装されているAgent
        MenzenAgent(),  # mjxに実装されているAgent
        MenzenAgent(),  # mjxに実装されているAgent
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

