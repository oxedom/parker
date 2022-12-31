from flask import Flask,request
import os
import cv2
import numpy as np
from flask_cors import CORS
import json
from pathlib import Path
import time
import base64

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)




app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app)
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app.config['SECRET_KEY'] = 'secret!'


def parseReqtoBuffer(reqObject):

    #Request object -> get Json
    server_object = reqObject.get_json()

    #Parse JSON to Python Object
    server_object_str = json.dumps(server_object)
    
    #Load buffer Object from buffer prop from object
    buffer_object = json.loads(server_object_str)['buffer']

    return buffer_object


@app.route('/cv2', methods=['POST'])
def handle_cv2():
    # serverObject with buffer props that contains RAW image buffer from client
    buffer_object = parseReqtoBuffer(request)

  
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
    faces_rect = haar_cascade.detectMultiScale(blurred, scaleFactor=2.1, minNeighbors=9)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, "Open CV", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    img_encode = cv2.imencode('.jpg', img)[1]
    # Converting the image into numpy array
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    im_b64 = base64.b64encode(byte_encode)
    im_b64_utf8 = im_b64.decode('utf-8')

    resObj = {
        "img":f"data:image/jpg;base64,{im_b64_utf8}",
        "meta data": {
            "faces": faces_rect
            
        }
        }
    # return f"data:image/jpg;base64,{im_b64_utf8}"
    return resObj
 



    # Convert the array buffer to a NumPy array
@app.route('/cv3', methods=['POST'])
def handle_cv3():

    buffer_object = parseReqtoBuffer(request)    
    # server_object = request.get_json('imageBuffer')

    # # Convert the dictionary object to a string
    # server_object_str = json.dumps(server_object)

    # buffer_object = json.loads(server_object_str)['buffer']

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
    blob = cv2.dnn.blobFromImage(img, 1/255.0 ,(320 ,320), swapRB=True, crop=False)
  
    r = blob[0, 0, :, :]
    r0 = blob[0, 0, :, :]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    net.setInput(blob)

    t0 = time.time()
    outputs = net.forward(output_layers)
    t = time.time()



    print('time=', t-t0)

    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                w = int(detection[2] * img_height)
                h = int(detection[3] * img_height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidenceLevel = str(round(confidences[i] * 100, 2)) + "%"
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, 255, 3)
            cv2.putText(img, confidenceLevel, (x, 50), font, 3, 255, 3)

 



    img_encode = cv2.imencode('.jpg', img)[1]
    # Converting the image into numpy array
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    im_b64 = base64.b64encode(byte_encode)
    im_b64_utf8 = im_b64.decode('utf-8')


    
    resObj = {
        "img":f"data:image/jpg;base64,{im_b64_utf8}",
        "meta data": {
            "boxes": boxes,
            "confidences": confidences,
        }
        }
    # return f"data:image/jpg;base64,{im_b64_utf8}"
    return resObj

    

if __name__ == '__main__':
    # socketio.run(app)
    app.run(host='0.0.0.0')


