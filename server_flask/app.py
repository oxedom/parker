from flask import Flask,request
# from flask_socketio import SocketIO, send
import cv2
import numpy as np
from flask_cors import CORS
import json
import array

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*")

# @socketio.on('buffer')
# def handle_message(buffer):
#       print(buffer)


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
    print(img)
  
    return 'PYTHON'
 
    # Convert the array buffer to a NumPy array
  
    # print('I got a buffer')
  
    # print(buffer)
    # cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
 
    # mat = np.asarray(buffer, dtype=np.uint8)     
    # socketio.emit('gray', cv2.imdecode(arr,'Grayscale'))

    

if __name__ == '__main__':
    # socketio.run(app)
    app.run()