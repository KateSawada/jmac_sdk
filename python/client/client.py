from __future__ import annotations

from typing import List
import threading
import time
import eventlet
import socketio
from datetime import datetime
import json
import random

import mjx
from mjx import agents


# SocketIOのテンプレート
class SocketIOClient:
    # namespaceの設定用クラス
    class NamespaceClass(socketio.ClientNamespace):
        def on_connect(self):
            pass
        def on_disconnect(self):
            pass
        def on_message(self, data):
            pass
        def on_server_to_client(self, data):
            pass
        def on_ask_act(self, data):
            pass
    # 接続時に呼ばれるイベント
    def on_connect(self):
        # print('Connected to server (%s:%d, namespace="%s", query="%s")',\
        #              self.ip_,self.port_,self.namespace_,self.query_)
        print("connect")
        self.is_connect_ = True

    # 切断時に呼ばれるイベント
    def on_disconnect(self):
        print('Disconnected from server (%s:%d, namespace="%s", query="%s")',\
                     self.ip_,self.port_,self.namespace_,self.query_)
        self.is_connect_ = False
    def on_message(self, data):
        print('Received message %s', str(data))
    # サーバーからイベント名「server_to_client」でデータがemitされた時に呼ばれる
    def on_server_to_client(self, data):
        print('Received message %s', str(data))

    def on_ask_act(self, data):
        """サーバーから次の行動を決定する要求があったときに呼ばれる

        Args:
            data (???): 盤面情報
        """
        # data = json.loads(data)
        data = mjx.Observation(data)
        decided_action = self.agent.act(data).to_json()
        return decided_action

    # Namespaceクラス内の各イベントをオーバーライド
    def overload_event(self):
        self.Namespace.on_connect          = self.on_connect
        self.Namespace.on_disconnect       = self.on_disconnect
        self.Namespace.on_message          = self.on_message
        self.Namespace.on_server_to_client = self.on_server_to_client
        self.Namespace.on_ask_act = self.on_ask_act
    # 初期化
    def __init__(self,ip,port,namespace,query,agent,room_id,player_name):
        self.ip_         = ip
        self.port_       = port
        self.namespace_  = namespace
        self.query_      = query
        self.is_connect_ = False
        self.sio_        = socketio.Client()
        self.Namespace   = self.NamespaceClass(self.namespace_)
        self.overload_event()
        self.sio_.register_namespace(self.Namespace)
        self.agent = agent
        self.room_id = room_id
        self.player_name = player_name
    # 接続確認
    def isConnect(self):
        return self.is_connect_
    # 接続
    def connect(self):
        # 接続先のURLとqueryの設定
        url = 'ws://'+self.ip_+':'+str(self.port_)+'?query='+self.query_
        print('Try to connect to server(%s:%d, namespace="%s", query="%s")',\
                     self.ip_,self.port_,self.namespace_,self.query_)
        try:
            self.sio_.connect(url, namespaces=self.namespace_)
        except:
            print('Cannot connect to server(%s:%d, namespace="%s", query="%s")',\
                          self.ip_,self.port_,self.namespace_,self.query_)
        else:
            if not self.is_connect_:
                print('Namespace may be invalid.(namespace="%s")',\
                              self.namespace_)
    # 切断
    def disconnect(self):
        try:
            self.sio_.disconnect()
        except:
            print('Cannot disconnect from server(%s:%d, namespace="%s", query="%s")',\
                          self.ip_,self.port_,self.namespace_,self.query_)
        else:
            self.is_connect_ = False
    # 再接続
    def reconnect(self):
        self.disconnect()
        time.sleep(5)
        self.connect()
    # クライアントからデータ送信(send)する
    def sendData(self, data):
        try:
            self.sio_.send(data, namespace=self.namespace_)
        except:
            print('Has not connected to server(%s:%d, namespace="%s", query="%s")',\
                          self.ip_,self.port_,self.namespace_,self.query_)
        else:
            print('Send message %s (namespace="%s")', str(data), self.namespace_)
    # 独自定義のイベント名「client_to_server」で、クライアントからデータ送信(emit)する
    def emitData(self, data):
        try:
            self.sio_.emit('client_to_server', data, namespace=self.namespace_)
        except:
            print('Has not connected to server(%s:%d, namespace="%s", query="%s")',\
                          self.ip_,self.port_,self.namespace_,self.query_)
        else:
            print('Emit message %s (namespace="%s")', \
                         str(data), self.namespace_)
    # メインの処理
    def run(self):
        while True:
            self.connect()
            time.sleep(1)
            if self.is_connect_:
                break
        p = threading.Thread(target=self.sio_.wait)
        p.setDaemon(True)
        p.start()

    def enter_room(self):
        data = {
            'room_id': self.room_id,
            'player_name': self.player_name,
        }
        self.sio_.emit('enter_room', data=data, namespace = self.namespace_, ) # ルームIDを指定して入室


class RandomAgent(mjx.Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions: List[mjx.Action] = observation.legal_actions()
        return random.choice(legal_actions)

if __name__ == '__main__':
    # Ctrl + C (SIGINT) で終了
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    # SocketIO Client インスタンスを生成
    agent = agents.ShantenAgent()
    room_id = 123  # NOTE: DEBUG
    sio_client = SocketIOClient('localhost', 5000, '/test', 'secret', agent, room_id)
    # SocketIO Client インスタンスを実行
    sio_client.run()
    sio_client.enter_room()
    # データを送信
    # for i in range(10):
    #     sio_client.sendData({'test_data': 'send_from_client'})
    #     sio_client.emitData({'test_data': 'emit_from_client'})
    #     time.sleep(1)
    # 切断
    # sio_client.disconnect()
    # print('Finish')
    # 終了
    # exit()
