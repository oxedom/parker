from flask import Flask
from flask_socketio import SocketIO, send
import cv2
import numpy as np

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('buffer')
def handle_message(buffer):
    # print('I got a buffer')
    # array = np.frombuffer(buffer, dtype=np.uint8)
    socketio.emit('gray', buffer)

    

if __name__ == '__main__':
    socketio.run(app)