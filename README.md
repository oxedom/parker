# Parkerr

Parkerr is a open source smart parking browser application that enables you monitor parking spots using a webcam, cellphone camera, or any virtual camera. The tool uses computer vision object detection to operate and all the computation is processed inside browser utilizing TensorFlow.js. Communication between a remote cellphone and the browser is enabled through WebRTC (PeerJS uses PeerServer for session metadata and candidate signaling. as well as Google Stun servers)

# LIVE LINK

[Live link](https://www.parkerr.org/)

![Parkerr Demo](https://i.imgur.com/JSEIqFD.png)

# Built With

- TensorFlow.js
- Tailwind
- NextJS
- PeerJS (WebRTC)
- YOLO7 (Original Model)
- YOLO7-tfjs (Ported Model)

# Mobile Phone Camera Instructions

Navigate to the "Vision" page, press the designated remote button, use your mobile device to scan the QR code. Once the page is loaded, locate and press the "Call" button and allow access to your phone's camera when prompted.

1. Make sure phone is streaming video in landscape mode
2. Change your phone settings that your phone screen doesn't autolock to ensure a continutes video stream.
3. You may need to press the call button a few times to establish the connection.

# How to Connect CCTV/IP Cameras that are not directly connected to Parker

1. Open OBS (you can download it from https://obsproject.com/).
2. Click on the "+" icon in the "Sources" box and select "Window Capture".
3. Choose the camera software window from the list of windows available for capture.
4. Click "OK" to confirm your selection.
5. Click Start Virtual Camera

You can adjust output resolution in the settings, recommended between 640x480 till 1280x720.

# How to Connect CCTV/IP Cameras that are on the local network.

If your iP/CCTV camera is on your local network, you can set your OBS settings to an ip address.

Youtube Guide:
https://www.youtube.com/watch?v=0z9Te51rh-4

# Webcam Instructions

Open the vision page, press the webcam button, allow webcam access, and point it wherever you desire.

# Settings Documentation

1. _Processing_: Toggle controls if the TFJS engine will process the video input
2. _Show Boxes_: Toggle the bounding boxes
3. _Vehicle Only_: Detect only vehicles; when switched off, bounding boxes can be occupied by any kind of detections
4. _Detection Threshold_: The detection score threshold
5. _IOU Threshold_: Non maximum suppression/Jaccard/IntersectionOverUnion threshold, the higher the more tolerant it is for colliding bboxes
6. _FPS: Render Rate_, the lower the faster the model detects image, fastest is 10 frames per secound, default is 1 frame per secound, max is 1 frame every 2 secound

# Can Parker be processed the on a server?

Yes, it can be processed on a server! Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo7, which worked fine. However, without a very good VPS with a GPU, the CPU can't handle many rendering requests. Functions were created in previous versions of Parkerr that encode the images into blobs to the server. The JSON response might need a bit of tweaking to be compatible, but just switch the process function with an API request. If you do, please make a PR and send a message!

`docker pull oxedom/flask_api` <br/>
`docker run -p 5000:5000 flask_api` <br/>
[Dockerhub Image](https://hub.docker.com/repository/docker/oxedom/flask_api/) <br/>

# What else can parker detect?

[Here's a list](https://github.com/oxedom/parker/blob/main/client/libs/labels.json)

# Can I Contribute?

Yes sure, submit a PR!
