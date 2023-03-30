import random

import mjx
import torch
from client.agent import CustomAgentBase
from client.client import SocketIOClient
from custom_ai import Net, Net2
from torch import Tensor, nn, optim, utils


# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．
class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        # self.model = torch.load('models/second_model.pth')
        self.model = Net2()
        self.model.load_state_dict(torch.load("models/fifth_model.pth"))

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        # // 0~33: Discard m1~rd
        # // 34,35,36: Discard m5(red), p5(red), s5(red)
        # // 37~70: Tsumogiri m1~rd
        # // 71,72,73: Tsumogiri m5(red), p5(red), s5(red)
        # // 74~94: Chi m1m2m3 ~ s7s8s9
        # // 95,96,97: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
        # // 98,99,100: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
        # // 101,102,103: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
        # // 104~137: Pon m1~rd
        # // 138,139,140: Pon m5(w/ red), s5(w/ red), p5(w/ red)
        # // 141~174: Kan m1~rd
        # // 175: Tsumo
        # // 176: Ron
        # // 177: Riichi
        # // 178: Kyuushu
        # // 179: No
        # // 180: Dummy

        legal_actions = obs.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 予測
        feature = obs.to_features(feature_name="mjx-small-v0")
        # feature = obs.to_features(feature_name="mjx-large-v0")
        self.model.eval()
        with torch.no_grad():
            action_logit = self.model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()

        # アクション決定
        mask = obs.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)


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
