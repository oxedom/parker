from flask import Flask,request
# from flask_socketio import SocketIO, send
import os
import cv2
import numpy as np
from flask_cors import CORS
import json
from pathlib import Path
import time


# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app.config['SECRET_KEY'] = 'secret!'





@app.route('/cv2', methods=['POST'])
def handle_cv2():
    # serverObject with buffer props that contains RAW image buffer from client
    server_object = request.get_json()

    # Convert the dictionary object to a string
    server_object_str = json.dumps(server_object)
    
    buffer_object = json.loads(server_object_str)['buffer']['data']

  
    #Convert the buffer_object from the JS to a bytearray
    data = bytearray(buffer_object)
    #Convert to to a npArray
    arr = np.frombuffer(data, np.uint8)
    #Decode npArray to Image
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Height Width and Depth of the Image
    (h, w, d) = img.shape
    #Greyscaling the Image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Mainting aspect Ratio on Grey Image and resizing it
    r = 300.0 / w
    dim = (300,int(h * r))
    resized = cv2.resize(gray_image, dim)

    #Bluring image so it to remove noise that confuses algos
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)


    #Algo for Detecting faces returns array of all XY cords of face [[106  48  50  50]]
    # 106,48 is pt1 top left, 48,50 is bottom right of Face
    faces_rect = haar_cascade.detectMultiScale(gray_image, scaleFactor=2.1, minNeighbors=9)
    
    print(faces_rect)
    # retval, buffer = cv2.imencode('.jpg', img)
    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(img, "Open CV", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.circle(img, (x, y), (2) , (255, 0, 0), -1)
            # cv2.imwrite('image.jpg', img)
            edged = cv2.Canny(gray_image, 30, 150)
            cv2.imshow("Edged", edged)
            cv2.imshow('Photo', img)
            cv2.waitKey(0)




    # print(img)
    
    return ''
 



    # Convert the array buffer to a NumPy array
@app.route('/cv3', methods=['POST'])
def handle_cv3():
        # serverObject with buffer props that contains RAW image buffer from client
    server_object = request.get_json()

    # Convert the dictionary object to a string
    server_object_str = json.dumps(server_object)

    buffer_object = json.loads(server_object_str)['buffer']['data']


    #Convert the buffer_object from the JS to a bytearray
    data = bytearray(buffer_object)
    #Convert to to a npArray
    arr = np.frombuffer(data, np.uint8)
    #Decode npArray to Image
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)


    #Image to Blob
    (img_height, img_width, img_depth) = img.shape
    img_ratio = 300.0 / img_width
    dim = (300,int(img_height * img_ratio))

    img = cv2.GaussianBlur(img, (3, 3), 0)
    blob = cv2.dnn.blobFromImage(img, 1/255.0 ,(img_width ,img_height), swapRB=True, crop=False)
  
    r = blob[0, 0, :, :]
    r0 = blob[0, 0, :, :]
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]





    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    print('time=', t-t0)

    def tracker2(x):
        confidence = x/100
        r = r0.copy()
        for output in np.vstack(outputs):
            if output[4] > confidence:
                x, y, w, h = output[:4]
                p0 = int((x-w/2)*img_height), int((y-h/2)*img_height)
                p1 = int((x+w/2)*img_width), int((y+h/2)*img_width)
                cv2.rectangle(r, p0, p1, 1, 1)
                cv2.imshow('blob', r)
                cv2.waitKey(0)   


    tracker2(50)


    return ''
  


    

if __name__ == '__main__':
    # socketio.run(app)
    app.run()


