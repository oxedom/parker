from flask import Flask,request
# from flask_socketio import SocketIO, send
import os
import cv2
import numpy as np
from flask_cors import CORS
import json
from pathlib import Path


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app.config['SECRET_KEY'] = 'secret!'

# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")



@app.route('/cv2', methods=['POST'])
def handle_cv():
    # serverObject with buffer props that contains RAW image buffer from client
    server_object = request.get_json()

    # Convert the dictionary object to a string
    server_object_str = json.dumps(server_object)
    
    buffer_object = json.loads(server_object_str)['buffer']['data']

    # array = np.frombuffer(bytearray(buffer_object), np.uint8)

    data = bytearray(buffer_object)

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=9)
    # retval, buffer = cv2.imencode('.jpg', img)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv2.imwrite('image.jpg', img)





    # print(img)
    
    return ''
 
    # Convert the array buffer to a NumPy array
  


    

if __name__ == '__main__':
    # socketio.run(app)
    app.run()


