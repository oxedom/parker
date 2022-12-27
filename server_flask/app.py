from flask import Flask,request
from flask_socketio import SocketIO, send
import cv2
import numpy as np
from flask_cors import CORS



app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*")

# @socketio.on('buffer')
# def handle_message(buffer):
#       print(buffer)


@app.route('/cv2', methods=['POST'])
def handle_cv():
    print(request.json)
    return 'LOGGED IN'



 
    # Convert the array buffer to a NumPy array
  
    # print('I got a buffer')
    # array = np.frombuffer(buffer, dtype=np.uint8)
    # print(buffer)
    # cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
 
    # mat = np.asarray(buffer, dtype=np.uint8)     
    # socketio.emit('gray', cv2.imdecode(arr,'Grayscale'))

    

if __name__ == '__main__':
    # socketio.run(app)
    app.run()