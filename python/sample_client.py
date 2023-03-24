import random

import mjx

from enum import IntEnum
from typing import List

from client.client import SocketIOClient
from client.agent import CustomAgentBase

# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．
class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.round=-1
        self.honba=0
        self.who=-1
        self.first_tsumo=True
        self.chitoitsu=False
        self.allow_pon=False
        self.jikaze=-1
        self.bakaze=-1
        self.yakujihai={}
        self.hiyakujihai={}
        self.yakuhai={}
        self.yaku=False
        self.pon_manzu=False
        self.pon_pinzu=False
        self.pon_sohzu=False
        self.turn =0
        self.done_betaori=False
        

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        legal_actions = obs.legal_actions()

        #if there's only one action, do it.
        if len(legal_actions) == 1:
            return legal_actions[0]

        if obs.round() > self.round or obs.honba() > self.honba:
            self.round=obs.round()
            self.honba=obs.honba()
            self.first_tsumo=True
            self.jikaze=(obs.who()-obs.round()) % 4
            self.bakaze=(obs.round()//4)%4
            self.yakujihai = {self.jikaze+27,self.bakaze+27 }
            self.hiyakujihai = {27,28,29,30}-self.yakujihai
            self.yakuhai = self.yakujihai | {31,32,33}
            self.yaku=False
            self.allow_pon=False
            self.pon_manzu=False
            self.pon_pinzu=False
            self.pon_sohzu=False
            self.done_betaori=True
            self.turn =0
            toitsunum=0
            curr_hand=obs.MjxLargeV0().current_hand(obs)
            for v in curr_hand[1]:
                toitsunum+=v
            if toitsunum==6:
                self.chitoitsu=True
            elif toitsunum >= 4:
                self.allow_pon=True
            


        
        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [mjx.ActionType.TSUMO, mjx.ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        if len(obs.curr_hand().closed_tiles()) in range(2,15,3):
            self.turn+=1
            #first tsumo
            if self.first_tsumo:
                self.first_tsumo=False
                # if kyushukyuhai, do ryukyoku
                nine_actions = [a for a in legal_actions if a.type() == mjx.ActionType.ABORTIVE_DRAW_NINE_TERMINALS]
                if len(nine_actions) >= 1:
                    assert len(nine_actions) == 1
                    return nine_actions[0]
            
            
            curr_hand=obs.MjxLargeV0().current_hand(obs)
            
            single = []
            toitsu = []
            kohtsu = []
            ankan = []
            discards = []

            for i in range(34):
                if curr_hand[0][i]==1 and curr_hand[1][i]==0:
                    single.append(i)
                elif curr_hand[1][i]==1 and curr_hand[2][i]==0:
                    toitsu.append(i)
                elif curr_hand[2][i]==1 and curr_hand[3][i]==0:
                    kohtsu.append(i)
                elif curr_hand[3][i]==1:
                    ankan.append(i)

            
            #CHITOITSU
            if len(toitsu) >= 6:
                self.chitoitsu=True
                discards = single
                return self.chosen_discard(obs,discards)
                
            elif len(toitsu) >= 5 and len(kohtsu)>=1:
                self.chitoitsu=True
                discards = kohtsu
                return self.chosen_discard(obs,discards)
            elif len(toitsu) >= 5 and len(ankan) >=1:
                self.chitoitsu=True
                discards = ankan
                return self.chosen_discard(obs,discards)
            
            
            for s in single:
                if s in self.hiyakujihai:
                    discards.append(s)
            if len(discards)>0:
                return self.chosen_discard(obs,discards)

            for s in single:
                if s in self.yakujihai:
                    discards.append(s)
            if len(discards)>0:
                return self.chosen_discard(obs,discards)

            #TOITOIHO
            
            if len(toitsu)>= 4:
                self.allow_pon=True
                discards=single
                return self.chosen_discard(obs,discards)

            jihai = []
            yaochuhai=[]
            manzu = []
            pinzu = []
            sohzu = []
            for v in obs.curr_hand().closed_tiles():
                if v in range(0,9):
                    manzu.append(v)
                    if v == 0 or v == 8:
                        yaochuhai.append(v)
                elif v in range(9,18):
                    pinzu.append(v)
                    if v == 9 or v == 17:
                        yaochuhai.append(v)
                elif v in range(18,27):
                    sohzu +=1
                    if v == 18 or v == 26:
                        yaochuhai.append(v)
                else:
                    jihai.append(v)

            if(self.turn<=6 or self.turn >=12):
                riichi_actions = [a for a in legal_actions if a.type() == mjx.ActionType.RIICHI]
                if len(riichi_actions) >= 1:
                    return riichi_actions[0]
            
            #HONITSU
            if (self.pon_manzu+self.pon_pinzu+self.pon_sohzu)<2:
                
                if(self.pon_manzu):
                    maxhai=len(manzu)
                    maxcol="m"
                elif(self.pon_pinzu):
                    maxhai=len(pinzu)
                    maxcol="p"
                elif(self.pon_sohzu):
                    maxhai=len(sohzu)
                    maxcol="s"
                else:
                    maxhai = max(len(manzu),len(pinzu),len(sohzu))
                    if(maxhai==len(manzu)):
                        maxcol="m"
                    elif(maxhai==len(pinzu)):
                        maxcol="p"
                    else:
                        maxcol="s"
                    

                tashoku = len(manzu)+len(pinzu)+len(sohzu) -maxhai
                
                if tashoku >=(5-self.turn//6):
                    
                    if maxcol !="m":
                        discards+=manzu
                    if maxcol !="p":
                        discards +=pinzu
                    if maxcol != "s":
                        discards+=sohzu
                    return self.chosen_discard(obs,discards)
            
            #MENTANPIN
            if len(obs.curr_hand().closed_tiles())==14:
                if self.is_effective(obs,jihai):
                    return self.chosen_discard(obs,jihai)
                return self.chosen_discard(obs,yaochuhai)

            return self.chosen_discard(obs,[])
        
        else:
            
            if not self.chitoitsu:
                pon_actions = [
                    a
                    for a in legal_actions
                    if a.type() == mjx.ActionType.PON
                ]
                if len(pon_actions)>=1:
                    assert len(pon_actions)==1
                    #役牌は鳴く
                    nakihai = pon_actions[0].open().stolen_tile().type()
                    if(nakihai in self.yakuhai):
                        self.yaku=True
                        return pon_actions[0]
                    elif(self.allow_pon):
                        if(nakihai in range(0,9)):
                            self.pon_manzu=True
                        elif(nakihai in range(9,18)):
                            self.pon_pinzu=True
                        elif(nakihai in range(18,27)):
                            self.pon_sohzu=True
                        return pon_actions[0]

            pass_action = [
                a
                for a in legal_actions
                if a.type() == mjx.ActionType.PASS
            ]
            return pass_action[0]

    def chosen_discard(self, obs:mjx.Observation, discards:List[int]) -> mjx.Action:
        
        legal_actions = obs.legal_actions()
        effective_discard_types = obs.curr_hand().effective_discard_types()
        legal_discards = [
            a for a in legal_actions if a.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]
        ]

        chosen_discards = [
            a for a in legal_discards if a.tile().type() in discards
        ]
        
        
        if len(chosen_discards) > 0:
            chosen_effective_discards =[
                a for a in chosen_discards if a.tile().type() in effective_discard_types
            ]
            
            
            if len(chosen_effective_discards) > 0:
                return random.choice(chosen_effective_discards)
            return random.choice(chosen_discards)
        
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]
        if len(effective_discards) > 0:
            return random.choice(effective_discards)
        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)    

    
if __name__ == "__main__":
    # 4人で対局する場合は，4つのSocketIOClientで同一のサーバーに接続する．
    my_agent = MyAgent()  # 参加者が実装したプレイヤーをインスタンス化

    sio_client = SocketIOClient(
        ip='localhost',
        port=5000,
        namespace='/test',
        query='secret',
        agent=my_agent,  # プレイヤーの指定
        room_id=123,  # 部屋のID．4人で対局させる時は，同じIDを指定する．
        )
    # SocketIO Client インスタンスを実行
    sio_client.run()
    sio_client.enter_room()
