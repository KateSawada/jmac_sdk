from mjx import ActionType, Agent, Observation, Action
import random

class MenzenAgent(Agent):
    """A rule-based agent, which plays just to reduce the shanten-number.
    The logic is basically intended to reproduce Mjai's ShantenPlayer.
    - Mjai https://github.com/gimite/mjai
    """

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

        # if it can apply chi/pon/open-kan, pass
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]

        # 鳴きができる場合、代わりにPASSを選択する
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