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


function decodeBase64(base64) {
  // Create a buffer to hold the binary data
  const buffer = new Uint8Array(base64.length * 3 / 4);

  // Iterate over the base64 characters
  for (let i = 0, j = 0; i < base64.length; i += 4) {
    // Convert each base64 character to a 6-bit binary value
    const c0 = base64.charCodeAt(i);
    const c1 = base64.charCodeAt(i + 1);
    const c2 = base64.charCodeAt(i + 2);
    const c3 = base64.charCodeAt(i + 3);

    // Convert the 6-bit binary values to 8-bit binary values
    buffer[j++] = (c0 << 2) | (c1 >> 4);
    buffer[j++] = ((c1 & 15) << 4) | (c2 >> 2);
    buffer[j++] = ((c2 & 3) << 6) | c3;
  }

  // Return the buffer as an ArrayBuffer
  return buffer.buffer;
}


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