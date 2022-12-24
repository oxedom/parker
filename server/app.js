const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server, {  cors: {
  origin: "http://localhost:3000",
  methods: ["GET", "POST"]
}});


function isValidBase64Image(str) {
  const regex = /^data:image\/(png|jpeg|jpg|gif|bmp|webp);base64,[A-Za-z0-9+/]+=*$/;
  return regex.test(str);
}


const cv = require('opencv4nodejs');
const cors = require('cors');




function decodeBase64ToMat(base64) {
  // Convert the base64-encoded string to a Uint8Array
  const data = new Uint8Array(atob(base64).split('').map(char => char.charCodeAt(0)));

  // Create an OpenCV Mat from the Uint8Array
  return cv.matFromArray(data, 1);
}


app.use(cors());


  io.on('connection', socket => {

    socket.on('stream', data => {

      
      const imageBuffer = Buffer.from(data, 'base64');
      // const image = cv.imdecode(imageBuffer)
      // const grayImage = image.cvtColor(cv.COLOR_BGR2GRAY);



      socket.emit('output', data);
      // socket.emit('matt', imageBuffer);
  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});