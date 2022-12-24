const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);
const cv = require('opencv4nodejs');



// Socket.io event for receiving webcam data from the React app
io.on('connection', socket => {
  socket.on('stream', data => {
    // Read the video data as an image
    const image = cv.imdecode(data);

    // Convert the image to grayscale
    const grayImage = image.cvtColor(cv.COLOR_BGR2GRAY);

    // Encode the image as a data URL
    const dataUrl = cv.imencode('.jpg', grayImage).toString('base64');

    // Send the output video to the React app using the "output_video" event
    socket.emit('output_video', `data:image/jpg;base64,${dataUrl}`);
  });
});

server.listen(3000);