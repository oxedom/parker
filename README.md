# Parker

Parker is an open-source parking browser application that enables you to monitor parking spots using a webcam, cellphone camera, or any virtual webcam. The tool uses computer vision object detection to operate, and all computation is processed inside the browser using TensorFlow.js. Communication between a remote cellphone and the browser is enabled through WebRTC (PeerJS uses PeerServer for session metadata and candidate signaling, as well as Google STUN servers).

# licensing

Under the MIT License.

# LIVE LINK

[Live link](https://parker-oxedoms-projects.vercel.app/)


# Built With

- Vue 3 + Vite + TypeScript
- Pinia (state)
- Vue Router
- Tailwind CSS
- TensorFlow.js
- PeerJS (WebRTC)
- YOLO7 (Original Model)
- YOLO7-tfjs (Ported Model)

# Run locally

```bash
cd client
npm install
npm run dev      # http://localhost:3000
npm run build    # type-check + production build to client/dist
npm run preview  # serve the production build
```

# Can Parker be processed the on a server?

Yes, it can be processed on a server, Before refactoring the architecture of the entire project, a flask API was built using OpenCV and Yolo3. It worked well but it wasn't scalable. Utils Functions were created in previous versions of Parker that encode the images into blobs to the server. The JSON response need a bit of tweaking to be compatible, but implementing the server processing would not require too much of a refactor in the client project. If anyone is in need of help open a issue and I will gladly assist.

`docker pull oxedom/flask_api` <br/>
`docker run -p 5000:5000 flask_api` <br/>
[Dockerhub Image](https://hub.docker.com/repository/docker/oxedom/flask_api/) <br/>

# What else can parker detect?

[Here's a list](https://github.com/oxedom/parker/blob/main/client/src/lib/labels.ts)

# Can I Contribute?

Yes sure, submit a PR!
