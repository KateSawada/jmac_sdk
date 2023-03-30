import random

import mjx
from mjx.const import ActionType
import numpy as np

from client.client import SocketIOClient
from client.agent import CustomAgentBase
from rule_based_agent import RuleAgent


if __name__ == "__main__":
    # 4人で対局する場合は，4つのSocketIOClientで同一のサーバーに接続する．
    my_agent = RuleAgent()  # 参加者が実装したプレイヤーをインスタンス化

    sio_client = SocketIOClient(
        ip='localhost',
        port=5000,
        namespace='/test',
        query='secret',
        agent=my_agent,  # プレイヤーの指定
        room_id=123,  # 部屋のID．4人で対局させる時は，同じIDを指定する．
        player_name='hogehoge' # プレイヤーの名前
        )
    # SocketIO Client インスタンスを実行
    sio_client.run()
    sio_client.enter_room()
