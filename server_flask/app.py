from flask import Flask
import base64
import cv2
import numpy as np
import socketio

app = Flask(__name__)
sio = socketio.Server()

@sio.on('stream')
def handle_stream(data):
  # decode the image data and convert it to a NumPy array
  image = base64.b64decode(data)
  image = np.frombuffer(image, dtype=np.uint8)
  # convert the image to a OpenCV image and display it
  image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
  cv2.imshow('image', image)
  # send the image back to the client
  image = cv2.imencode('.jpg', image)[1]
  image = base64.b64encode(image).decode()
  sio.emit('output', image)

if __name__ == '__main__':
  sio.run(app, host='localhost', port=5000)