from server.server import SocketIOServer


if __name__ == "__main__":

    sio_server = SocketIOServer(
        ip='localhost',
        port=5000,
        namespace='/test',
        logging=True,  # ログを出力するか否か
        is_solo=False  # 1人で対局をするか，4人で対局するか
        )
    sio_server.start()
