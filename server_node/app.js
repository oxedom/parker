

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


// const cv = require('opencv4nodejs');
const cors = require('cors');
const { log } = require('console');





app.use(cors());


  io.on('connection', socket => {
    console.log('Connection made')
    socket.on('stream', data => {

    //   const faceClassifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_DEFAULT)
    //   const base64Image =  data.split(';base64,').pop();
     
    //   const frame = cv.imdecode(Buffer.from(base64Image, 'base64'));
    //   const faces = faceClassifier.detectMultiScale(frame).objects;
    //   console.log(faces);
    
    //     const blackAndWhiteFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
    //     const resized_blackAndWhiteFrame = blackAndWhiteFrame.resize(100,100)
    //     const outputData = cv.imencode('.jpg', blackAndWhiteFrame).toString('base64');

    //     const final = `data:image/jpeg;base64,${outputData}`
        

      socket.emit('output', data);
  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});