import random
import torch

import mjx

from client.client import SocketIOClient
from client.agent import CustomAgentBase

from model.cnn import CNN_MLP

# モデルの読み込み
# state_dict()はパラメータのみを保存するため、モデル構造を定義してから読み込む
model = CNN_MLP()
model.load_state_dict(torch.load('./model/model.pth'))

# CustomAgentBase を継承して，
# custom_act()を編集して麻雀AIを実装してください．
class MyAgent(CustomAgentBase):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．

        Args:
            obs (mjx.Observation): 盤面情報と取れる行動(obs.legal_actions())

        Returns:
            mjx.Action: 実際に取る行動
        """
        legal_actions = obs.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # リーチできるならリーチする
        riichi_actions = [a for a in legal_actions if a.type() == mjx.const.ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]
        
        # 予測
        feature = torch.Tensor(obs.to_features(feature_name="mjx-small-v0").ravel())
        with torch.no_grad():
            action_logit = self.model.predict(torch.Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()

        # アクション決定
        mask = obs.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)


if __name__ == "__main__":
    # 4人で対局する場合は，4つのSocketIOClientで同一のサーバーに接続する．
    my_agent = MyAgent(model)  # 参加者が実装したプレイヤーをインスタンス化

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
