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
const { drawDetection } = require('opencv4nodejs');




function decodeBase64ToMat(base64) {
  // Convert the base64-encoded string to a Uint8Array
  const data = new Uint8Array(atob(base64).split('').map(char => char.charCodeAt(0)));

  // Create an OpenCV Mat from the Uint8Array
  return cv.matFromArray(data, 1);
}


app.use(cors());


  io.on('connection', socket => {

    socket.on('stream', data => {

      // const faceClassifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_DEFAULT)
      // const base64Image =  data.split(';base64,').pop();
     
      // const frame = cv.imdecode(Buffer.from(base64Image, 'base64'));
      // const faces = faceClassifier.detectMultiScale(frame).objects;
      // faces.forEach(face => {

        
      //   const faceROI = frame.getRegion(face)
      //   frame = frame.drawCircle(1,2)

        // cv.rectangle( faceROI, new cv.Point(0, 0), new cv.Point(faceROI.cols - 1, faceROI.rows - 1), new cv.Vec(255, 0, 0), 2);
        


      // })
  
    // const blackAndWhiteFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
    // const outputData = cv.imencode('.jpg', blackAndWhiteFrame).toString('base64');

    // const final = `data:image/jpeg;base64,${outputData}`
    

      socket.emit('output', data);
      // socket.emit('matt', final);
  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});