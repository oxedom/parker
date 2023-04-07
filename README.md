
# Parker 
Parker is a free smart parking tool that enables you monitor parking spots using a webcam, cellphone camera, or any virtual camera. The tool uses computer vision object detection to process all the footage in the browser, utilizing TensorFlow.js. Communication between a remote cellphone and the browser is enabled through WebRTC (PeerJS uses PeerServer for session metadata and candidate signaling. as well as Google Stun servers)

# LIVE LINK
[Live link](https://www.parkerr.org/)

![Parker Demo](https://i.imgur.com/XCDlxNg.gif)

# Built With
* TensorFlow.js
* Tailwind
* NextJS
* PeerJS (WebRTC)
* YOLO7 (Original Model)
* YOLO7-tfjs (Ported Model)




# Mobile Phone Camera Instructions
After entering the vision page and pressing the remote stream button, you can simply scan QR code and after the page loads press on the call button to allow acesss to your phones camera, two imporant notes that the phone should be in landscape mode and change your auto-lock settings that your phone screen doesn't turn off after a mintue (Should be on never if you are inteended to use it for a long period of time). 


# How to Connect CCTV/IP Cameras that are not directly connected to Parker
If your iP/CCTV cameras not are usally not directly connected to your computer, you can stream their video footage using [OBS](https://obsproject.com/) Window Capture feature and create a virtual webcam on your PC. 

# How to Connect CCTV/IP Cameras that are on the local netork.
If your iP/CCTV camera is on your local network, you can set your OBS settings to an ip address.

Quite simple.
https://www.youtube.com/watch?v=0z9Te51rh-4



# Webcam Instructions
Open the vision page, press the webcam button, allow webcam access, and point it wherever you desire.

# Settings Documentation
1. Processing: Toggle controls if the TFJS engine will process the video input
2. Show Boxes: Toggle the bounding boxes
3. Vehicle Only: Detect only vehicles; when switched off, bounding boxes can be occupied by any kind of detections
4. Detection Threshold: The detection score threshold
5. IOU Threshold: Non maximum suppression/Jaccard/IntersectionOverUnion threshold, the higher the more tolerant it is for colliding bboxes 
6. FPS: Render Rate, the lower the faster the model detects image, fastest is 10 frames per secound, default is 1 frame per secound, max is 1 frame every 2 secound# Can Parker be processed the on a server? 
Yes, it can be processed on a server! 
There is even a docker image for it that runs YOLO4 with openCV

# Can Parker be processed the on a server? 

`docker pull oxedom/flask_api` <br/>
`docker run -p 5000:5000 flask_api` <br/>
[Dockerhub Image](https://hub.docker.com/repository/docker/oxedom/flask_api/) <br/>

Yes, it can be processed on a server! Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo7, which worked fine. However, without a very good VPS with a GPU, the CPU can't handle many rendering requests. Functions were created in previous versions of Parker that encode the images into blobs to the server. The JSON response might need a bit of tweaking to be compatible, but just switch the process function with an API request. If you do, please make a PR and send a message!




Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo7, which worked fine. However, without the right Infrastructure on the server (GPU and autoscaling) a CPU won't be able to handle many POST requests. Client side Functions were created in previous versions of Clients Parker that encode the webcam images into blobs int arrays to be posted server, code will need a refactroring to make it work with the modern version of parker. 
If someone is truly interested create an issue and I'll try and help out.



# What else can parker detect?
[Here's a list](https://github.com/oxedom/parker/blob/main/client/libs/labels.json)

# Can I Contribute?
Yes sure, submit a PR!
