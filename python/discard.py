# discard best tile

from search_suzi import *

yaotyu = [0,8,9,17,18,26,27,28,29,30,31,32,33]
zihai = [27,28,29,30,31,32,33]
def discard_effective(hand_discards,doras,remaining_tiles) -> mjx.Action:
            if len(hand_discards)==1: # 選択のしようがない
                return hand_discards[0]
            discard_effective_list = [a.tile().type() for a in hand_discards]
            best_effective_discard = hand_discards[0]

            effective_list = [1 for _ in range(34)]
            for i in discard_effective_list:
                effective_tiles = check_effective_hai(i)
                for j in effective_tiles:
                    bonus = 1
                    if j in doras:
                        bonus *= 2.5
                    if len(effective_tiles)==5 and (j==i-1 or j==i+1):
                        bonus *= 1.4
                    if remaining_tiles[0][j]==4:
                        effective_list[i] *= 1.8*bonus
                    elif remaining_tiles[0][j]==3:
                        effective_list[i] *= 1.6*bonus
                    elif remaining_tiles[0][j]==2:
                        effective_list[i] *= 1.4*bonus
                    elif remaining_tiles[0][j]==1:
                        effective_list[i] *= 1.2*bonus
                    elif remaining_tiles[0][j]==0:
                        if (i in [0,9,18]) and (j==i+1 or j==i+2):
                            effective_list[i] *= 0.1
                        elif (i in [1,10,19]) and (j==i+1):
                            effective_list[i] *= 0.1
                        elif (i in [1,10,19]) and (j==i+2):
                            effective_list[i] *=0.4*bonus
                        elif (i in [8,17,26]) and (j==i-1 or j==i-2):
                            effective_list[i] *= 0.1
                        elif (i in [7,16,25]) and (j==i-1):
                            effective_list[i] *= 0.1
                        elif (i in [7,16,25]) and (j==i-2):
                            effective_list[i] *= 0.4*bonus
                        else:
                            effective_list[i] *= 0.8*bonus
            
            # 最も効果的でない（最も不必要な）牌を選択
            effective_min = 100000
            for a in hand_discards:
                if effective_list[a.tile().type()]<effective_min:
                    effective_min = effective_list[a.tile().type()]
                    best_effective_discard = a
            return best_effective_discard

def discard_in_riichi(who,discards,hand_discards,dealer,doras,remaining_tiles,remaining_tile_num) -> mjx.Action:
            if len(hand_discards)==1: # 選択のしようがない
                return hand_discards[0]
            danger_point = {a:0 for a in range(34)}

            danger_list_1 =[1 for _ in range(35)]
            danger_list_2 =[1 for _ in range(35)]
            danger_list_3 =[1 for _ in range(35)]
            adjust_danger_list =[1 for _ in range(34)]

            for dora in doras:
                danger_list_1[dora]+=0.5
                danger_list_2[dora]+=0.5
                danger_list_3[dora]+=0.5

            if who[1][0]==1: # 下家リーチ
                danger_list_1[34] = 8
                if remaining_tile_num > 35:
                    for i in range(27,34):
                        danger_list_1[i] *= 1.5
            else:
                for i in range(27,34):
                    danger_list_1[i] /= 2 # 字牌の危険度を一律下げる
            discard_list_1 = [0 for _ in range(34)]
            for i in range(3,6): # 下家の捨て牌
                for j in range(34): # 牌をすべて探索
                    if discards[i][j]==1:
                        discard_list_1[j] = 1
            for i in range(34):
                if discard_list_1[i]==1: # 下家が捨てている牌
                    danger_list_1[i] = 0 # 牌の危険度を下げる(安全牌)
                    if i not in zihai:
                        for suzi in check_suzi(i):
                            if suzi in yaotyu:
                                danger_list_1[suzi] /= 4 # 牌の危険度を下げる(スジandヤオ九牌)
                            else:
                                danger_list_1[suzi] /= 2 # 牌の危険度を下げる(スジ)

            if who[2][0]==1: # 対面リーチ
                danger_list_2[34] = 8
                if remaining_tile_num > 35:
                    for i in range(27,34):
                        danger_list_2[i] *= 1.5
            else:
                for i in range(27,34):
                    danger_list_2[i] /= 2 # 字牌の危険度を一律下げる
            discard_list_2 = [0 for _ in range(34)]
            for i in range(6,9): # 対面の捨て牌
                for j in range(34): # 牌をすべて探索
                    if discards[i][j]==1:
                        discard_list_2[j] = 1
            for i in range(34):
                if discard_list_2[i]==1: # 対面が捨てている牌
                    danger_list_2[i] = 0 # 牌の危険度を下げる(安全牌)
                    if i not in zihai:
                        for suzi in check_suzi(i):
                            if suzi in yaotyu:
                                danger_list_2[suzi] /= 4 # 牌の危険度を下げる(スジandヤオ九牌)
                            else:
                                danger_list_2[suzi] /= 2 # 牌の危険度を下げる(スジ)

            if who[3][0]==1: # 上家リーチ
                danger_list_3[34] = 8
                if remaining_tile_num > 35:
                    for i in range(27,34):
                        danger_list_3[i] *= 1.5
            else:
                for i in range(27,34):
                    danger_list_3[i] /= 2 # 字牌の危険度を一律下げる
            discard_list_3 = [0 for _ in range(34)]
            for i in range(9,12): # 上家の捨て牌
                for j in range(34): # 牌をすべて探索
                    if discards[i][j]==1:
                        discard_list_3[j] = 1
            for i in range(34):
                if discard_list_3[i]==1: # 上家が捨てている牌
                    danger_list_3[i] = 0 # 牌の危険度を下げる(安全牌)
                    if i not in zihai:
                        for suzi in check_suzi(i):
                            if suzi in yaotyu:
                                danger_list_3[suzi] /= 4 # 牌の危険度を下げる(スジandヤオ九牌)
                            else:
                                danger_list_3[suzi] /= 2 # 牌の危険度を下げる(スジ)

            for i in range(34):
                if (discard_list_1[i]==1 and discard_list_2[i]==1 and discard_list_3[i]==1):
                    adjust_danger_list[i]=0 # 完全安牌
            
            for i in range(27,34):
                if remaining_tiles[0][i] == 1:
                    adjust_danger_list[i] = 0.1
                elif remaining_tiles[0][i] == 0:
                    adjust_danger_list[i] = 0
            
            if dealer==1:
                danger_list_1[34] *= 1.3
            elif dealer==2:
                danger_list_2[34] *= 1.3
            elif dealer==3:
                danger_list_3[34] *= 1.3

            danger_sum = [(x*danger_list_1[34]+y*danger_list_2[34]+z*danger_list_3[34])*w for (x,y,z,w) in zip(danger_list_1[:34],danger_list_2[:34],danger_list_3[:34],adjust_danger_list)]
            for i in range(34):
                danger_point[i] = danger_sum[i]

            # 最も安全な牌を選択
            danger_min = 1000000000
            danger_min_action = hand_discards[0]
            for a in hand_discards:
                if danger_point[a.tile().type()]<danger_min:
                    danger_min = danger_point[a.tile().type()]
                    danger_min_action = a
            
            return danger_min_action