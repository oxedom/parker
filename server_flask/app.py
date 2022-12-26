from flask import Flask
from flask_socketio import SocketIO,emit
# import cv2
# import numpy as np
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# def isValidBase64Image(str):
#     regex = /^data:image\/(png|jpeg|jpg|gif|bmp|webp);base64,[A-Za-z0-9+/]+=*$/
#     return regex.test(str)

@SocketIO.on('connect',)
def on_connect():
    print("Client connected")


@SocketIO.on('disconnect')
def on_disconnect():
    print("Client disconnected")

@SocketIO.on('message')
def on_message(message):
    print("Message", message)
    emit('reply', message)

# @SocketIO.on('connection')
# def handle_connection():
#     @socketio.on('stream')
#     def handle_stream(data):
#         emit('output', data)    

app = Flask(__name__)


# if __name__ == '__main__':
app.run()