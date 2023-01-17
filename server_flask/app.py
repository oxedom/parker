from flask import Flask,request
import os
import cv2
import numpy as np
from flask_cors import CORS
import json
from pathlib import Path
import time
import base64
from shapely import Polygon
# Load names of classes and get random colors
classes = open('engine/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('engine/yolov3.cfg', 'engine/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

app = Flask(__name__, static_folder='../build', static_url_path='/')
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app)
# haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app.config['SECRET_KEY'] = 'secret!'


def parseReqtoBuffer(reqObject):

    #Request object -> get Json
    server_object = reqObject.get_json()

    #Parse JSON to Python Object
    server_object_str = json.dumps(server_object)
    
    #Load buffer Object from buffer prop from object
    buffer_object = json.loads(server_object_str)['buffer']

    return buffer_object

def CalPercent(containerArea,intersectionArea):
    percent = 0
    if(intersectionArea > 0):
        percent = intersectionArea / containerArea
    return percent
        



def checkIntersection(rect1,rect2):
    rect1_top_y = rect1["cords"]['top_y']
    rect1_bottom_y = rect1["cords"]['bottom_y']
    rect1_left_x = rect1["cords"]['left_x']
    rect1_right_x = rect1["cords"]['right_x']
    
    rect2_top_y = rect2["cords"]['top_y']
    rect2_bottom_y = rect2["cords"]['bottom_y']
    rect2_left_x = rect2["cords"]['left_x']
    rect2_right_x = rect2["cords"]['right_x']

    x5 = max(rect1_right_x, rect2_right_x);
    y5 = max(rect2_top_y, rect1_top_y);
    x6 = min(rect1_left_x, rect2_left_x);
    y6 = min(rect1_bottom_y, rect2_bottom_y);
        
    xWidth = x6-x5
    xHeight = y6-y5
   
    containerArea= (rect1['dect_width']*rect1['dect_height'])
    print(containerArea)
    intersectionArea = xWidth*xHeight
    per = CalPercent(containerArea,intersectionArea)
    print(per)
    




    
@app.route('/api/cv/yolo_classes', methods=['GET'])
def handle_classes():
    return {"classes": classes}

@app.route('/api/cv/yolo', methods=['POST'])
def handle_yolo():


    buffer_object = parseReqtoBuffer(request)    

    #Convert the buffer_object from the JS to a bytearray
    data = bytearray(buffer_object)
    
    #Convert to to a npArray
    arr = np.frombuffer(data, np.uint8)
    #Decode npArray to Image
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    

    #Image to Blob
    (img_height, img_width, img_depth) = img.shape
   
    img_ratio = 300.0 / img_width
    # dim = (300,int(img_height * img_ratio))


    blob = cv2.dnn.blobFromImage(img, 1/255.0 ,(320 ,320), swapRB=True, crop=False)
    # test_height = img.shape;



    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    t = time.time()
    net.setInput(blob)

    outputs = net.forward(output_layers)
    t0 = time.time()
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
    detections = []
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidenceLevel = str(round(confidences[i] * 100, 2)) + "%"
            color = colors[i]

            detections.append(
                {
                    "dect_height": h, 
                    "dect_width": w,
                    "label": label,
                    "confidenceLevel": confidenceLevel,
                    "cords": { "right_x": x, "top_y": y, "width": w, "height": h}
                })
  

            cv2.rectangle(img, (x, y), (x + w, y + h), 200, 3)
            # cv2.putText(img, confidenceLevel, (x, 50), font, 3, 255, 3)
            cv2.putText(img, label, (x-20, 50), font, 3, 255, 3)

  
    img_encode = cv2.imencode('.jpg', img)[1]
    # Converting the image into numpy array
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    im_b64 = base64.b64encode(byte_encode)
    im_b64_utf8 = im_b64.decode('utf-8')



    resObj = {
        # "img":f"data:image/jpg;base64,{im_b64_utf8}",
        "time": time.time(),
        "img_width": img_width,
        "img_height": img_height,
        "meta_data": {
            "detections": detections,
            # "rawBase64": im_b64_utf8 
        },


        }
    

    return resObj


if __name__ == '__main__':
    app.run()




