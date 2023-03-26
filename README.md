# Parker 
Parker is a free smart parking tool that lets you monitor parking spots using a webcam, cellphone camera, or virtual camera. The tool uses computer vision object detection to process all the footage in the browser, utilizing TensorFlow.js. Communication between a remote cellphone and the browser is enabled through WebRTC, while PeerJS uses PeerServer for session metadata and candidate signaling.

# Built With
* TensorFlow.js
* NextJS
* PeerJS (WebRTC)
* YOLO7 (Original Model)
* YOLO7-tfjs (Ported Model)




# Mobile Phone Camera Instructions
After entering the vision page and pressing the remote stream button, you can simply scan QR code and after the page loads press on the call button to allow acesss to your phones camera, two imporant notes that the phone should be in landscape mode and change your settings that your phone screen doesn't turn off after a mintue. 


# How to Connect CCTV/IP Cameras
As IP/CCTV cameras are not directly connected to your computer, you can stream their video footage using [OBS](https://obsproject.com/) Window Capture feature and create a virtual webcam on your PC.


# Webcam Instructions
Open the vision page, press the webcam button, allow webcam access, and point it wherever you desire.

# Settings Documentation
1. Processing: Toggle controls if the TFJS engine will process the video input
2. Show Boxes: Toggle the bounding boxes
3. Vehicle Only: Detect only vehicles; when switched off, bounding boxes can be occupied by any kind of detections
4. Detection Threshold: The detection score threshold
5. IOU Threshold: Non maximum suppression/Jaccard/IntersectionOverUnion threshold, the higher the more tolerant it is for colliding bboxes 
6. FPS: Render Rate, the lower the faster the model detects image, fastest is 10 frames per secound, default is 1 frame per secound, max is 1 frame every 2 secounds
# Can Parker be processed the on a server? 
Yes, it can be processed on a server! Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo7, which worked fine. However, without a very good VPS with a GPU, the CPU can't handle many rendering requests. Functions were created in previous versions of Parker that encode the images into blobs to the server. The JSON response might need a bit of tweaking to be compatible, but just switch the process function with an API request. If you do, please make a PR and send a message!


# What else can parker detect?
[Here's a list](https://github.com/oxedom/parker/blob/main/client/libs/labels.json)

# Can I Contribute?
Yes, you can! Submit a PR.
