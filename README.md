# Parkerr

Parkerr is a open source smart parking browser application that enables you monitor parking spots using a webcam, cellphone camera, or any virtual camera. The tool uses computer vision object detection to operate and all the computation is processed inside browser utilizing TensorFlow.js. Communication between a remote cellphone and the browser is enabled through WebRTC (PeerJS uses PeerServer for session metadata and candidate signaling. as well as Google Stun servers)

# licensing 
Under the MIT License.

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

# Can Parker be processed the on a server?

Yes, it can be processed on a server! Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo3, which worked fine. However, without a very good VPS with a GPU, the CPU can't handle many rendering requests. Functions were created in previous versions of Parkerr that encode the images into blobs to the server. The JSON response might need a bit of tweaking to be compatible, but just switch the process function with an API request. If you do, please make a PR and send a message!

`docker pull oxedom/flask_api` <br/>
`docker run -p 5000:5000 flask_api` <br/>
[Dockerhub Image](https://hub.docker.com/repository/docker/oxedom/flask_api/) <br/>

# What else can parker detect?

[Here's a list](https://github.com/oxedom/parker/blob/main/client/libs/labels.json)

# Can I Contribute?

Yes sure, submit a PR!
