from __future__ import annotations

import threading
import time
import os
import eventlet
import socketio
from datetime import datetime
import json
import random

import mjx

from server import convert_log


class SocketIOServer:
    player_names_to_idx ={
        "player_0": 0,
        "player_1": 1,
        "player_2": 2,
        "player_3": 3,
    }

    # namespaceの設定用クラス
    class NamespaceClass(socketio.Namespace):
        def on_connect(self, sid, environ):
            pass
        def on_disconnect(self, sid):
            pass
        def on_message(self, sid, data):
            pass
        def on_client_to_server(self, sid, data):
            pass
        def on_enter_room(self, sid, room_id):
            pass
    # 接続時に呼ばれるイベント
    def on_connect(self, sid, environ):
        # print('Connected to client (sid: %s, environ: %s)',\
        #              str(sid), str(environ))
        print("connect")
        self.is_connect_ = True

    # 切断時に呼ばれるイベント
    def on_disconnect(self, sid):
        # print('Disconnected from client (sid: %s)',str(sid))
        print("disconnect")
        self.is_connect_ = False

    # サーバーからデータ送信(send)する
    def send_data(self, data):
        try:
            self.sio_.send(data, namespace=self.namespace_)
        except:
            print('Has not connected to client')
        else:
            print('Send message %s (namespace="%s")',\
                         str(data), self.namespace_)

    # 独自定義のイベント名「server_to_client」で、サーバーからデータ送信(emit)する
    def emit_data(self, data):
        try:
            self.sio_.emit('server_to_client', data, namespace=self.namespace_)
        except:
            print('Has not connected to client')
        else:
            print('Emit message %s (namespace="%s")',\
                         str(data), self.namespace_)

    def on_message(self, sid, data):
        print('Received message %s', str(data))
        self.send_data({'test_ack': 'send_from_server'})
    # クライアントからイベント名「on_client_to_server」でデータがemitされた時に呼ばれる
    def on_client_to_server(self, sid, data):
        print('Received message %s', str(data))
        self.emit_data({'test_ack': 'emit_from_server'})

    def on_enter_room(self, sid, room_id):
        """クライアントがルームに入った時の処理

        Args:
            sid (str): client sid
            room_id (str): room id
        """
        self.Namespace.enter_room(sid, room_id)
        if room_id not in self.clients.keys():
            self.clients[room_id] = []
        self.clients[room_id].append(sid)
        if (len(self.clients[room_id]) == self.n_start_players):
            self.envs[room_id] = mjx.MjxEnv()
            self.play(room_id)  # TODO: バックグラウンド処理に流したい

    def play(self, room_id):
        """
        Args:
            roomId: int
            server: SocketIOServer
        """
        print("game has started")
        # プレイヤーの位置を決める (ランダム)
        players = random.sample(self.clients[room_id], len(self.clients[room_id]))

        # 卓の初期化
        env_ = self.envs[room_id]
        obs_dict = env_.reset()
        logs = convert_log.ConvertLog()

        while not env_.done():
            actions = {}
            for player_id, obs in obs_dict.items():
                if self.is_solo:
                    sid = players[0]
                else:
                    sid = players[self.player_names_to_idx[player_id]]

                decided_action = self.Namespace.call('ask_act', obs.to_json(), to=sid, namespace='/test')
                actions[player_id] = mjx.Action(decided_action)
            obs_dict = env_.step(actions)
            if len(obs_dict.keys())==4:
                logs.add_log(obs_dict)
        returns = env_.rewards()
        if self.logging:
            self.save_log(obs_dict, env_, logs)
        print("game has ended")
        self.clients.pop(room_id)
        self.envs.pop(room_id)
        exit()
        
    # 対局終了時にログを保存
    def save_log(self, obs_dict, env, logs):
        logdir = "logs"
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        now = datetime.now().strftime('%Y%m%d%H%M%S')

        os.mkdir(os.path.join(logdir, now))
        for player_id, obs in obs_dict.items():
            with open(os.path.join(logdir, now, f"{player_id}.json"), "w") as f:
                json.dump(json.loads(obs.to_json()), f)
            with open(os.path.join(logdir, now, f"tenho.log"), "w") as f:
                f.write(logs.get_url())
        env.state().save_svg(os.path.join(logdir, now, "finish.svg"))

    # Namespaceクラス内の各イベントをオーバーライド
    def overload_event(self):
        self.Namespace.on_connect = self.on_connect
        self.Namespace.on_disconnect = self.on_disconnect
        self.Namespace.on_message = self.on_message
        self.Namespace.on_client_to_server = self.on_client_to_server
        self.Namespace.on_enter_room = self.on_enter_room
    # 初期化
    def __init__(self,ip,port,namespace,is_solo,logging=True):
        self.ip_          = ip
        self.port_        = port
        self.namespace_   = namespace
        self.is_connect_  = False
        self.sio_         = socketio.Server(async_mode='eventlet')
        self.app_         = socketio.WSGIApp(self.sio_)
        self.Namespace    = self.NamespaceClass(self.namespace_)
        self.logging = logging
        self.is_solo = is_solo
        self.n_start_players = 1 if is_solo else 4
        self.overload_event()
        self.sio_.register_namespace(self.Namespace)
        self.clients = {}
        self.envs = {}
    # 接続確認
    def isConnect(self):
        return self.is_connect_
    # 切断
    def disconnect(self,sid):
        try:
            self.sio_.disconnect(sid,namespace=self.namespace_)
        except:
            print('Cannot disconnect from Client(sid="%s", namespace="%s")',\
                          namespace=self.namespace_)
        else:
            self.is_connect_ = False
    # 開始
    def start(self):
        try:
            print('Start listening(%s:%d, namespace="%s")',\
                         self.ip_,self.port_,self.namespace_)
            eventlet.wsgi.server(eventlet.listen((self.ip_, self.port_)), self.app_)
        except:
            print('Cannot start listening(%s:%d, namespace="%s")',\
                          self.ip_,self.port_,self.namespace_)
    # メインの処理
    def run(self):
        p = threading.Thread(target=self.start)
        p.setDaemon(True)
        p.start()

    def start(self):
        app = socketio.WSGIApp(self.sio_) # wsgiサーバーミドルウェア生成
        # self.sio_.start_background_task(self.actively_send_json, self.room_id) # バックグラウンドタスクの登録
        eventlet.wsgi.server(eventlet.listen((self.ip_, self.port_)), app) # wsgiサーバー起動



if __name__ == '__main__':
    # Ctrl + C (SIGINT) で終了
    # SocketIO Server インスタンスを生成
    sio_server = SocketIOServer('localhost', 5000, '/test', logging=True, is_solo=False)
    sio_server.start()
    # SocketIO Server インスタンスを実行
    # sio_server.run()
    # 接続待ち
    # while not sio_server.isConnect():
    #     time.sleep(1)
    # 切断待ち
    # while sio_server.isConnect():
    #     time.sleep(1)
    # print('Finish')
    # 終了
    # exit()
