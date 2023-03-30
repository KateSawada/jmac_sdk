import mjx
import torch
from torch import Tensor
from abc import abstractmethod



class CustomAgentBase(mjx.Agent):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """参加者はこの関数をオーバーライドして行動を実装する"""
        pass

    def act(self, observation: mjx.Observation) -> mjx.Action:
        # 参加者が実装した関数でエラーが起きた場合は
        # - ツモ時: ツモ切り
        # - ポンとかの選択時: パス
        # をする
        try:
            return self.custom_act(observation)
        except:
            legal_actions = observation.legal_actions()
            if len(legal_actions) == 1:
                return legal_actions[0]
            for action in legal_actions:
                if action.type() in [mjx.ActionType.TSUMOGIRI, mjx.ActionType.PASS]:
                    return action

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
        if len(legal_actions == 1):
            return legal_actions[0]
        
        # 予測
        feature = obs.to_features(feature_name="mjx-small-v0")
        with torch.no_grad():
            action_logit = self.model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()

        # アクション決定
        mask = obs.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)