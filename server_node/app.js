

const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io_client = require('socket.io-client');
const cv = require('opencv4nodejs');

const io = require('socket.io')(server, {  cors: {
  origin: "http://localhost:3000",
  methods: ["GET", "POST"]
}});

// const python_socket = io_client('http://localhost:5000')

const cors = require('cors');


const { default: axios } = require('axios');

// python_socket.connect()
//CORS
app.use(cors());


  io.on('connection', socket => {

    console.log('Connection made')
    socket.on('stream', async imageBuffer => {
  

      const decoded_buffer = await cv.imdecodeAsync(imageBuffer)


      const payload = new Response(JSON.stringify('Hello, world!'), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }})

      try {
        const response = await axios.post('http://localhost:5000/cv2', { message: imageBuffer} )
      } catch (error) {
        
      }
   
      // python_socket.emit('buffer', decoded_buffer)
      console.log(response.data);
      
     

    //  python_socket.on('gray', data => {
    //   console.log(data);
  
    //  })
     //Detect faces
    // const faces = faceClassifier.detectMultiScale(frame).objects;

    
    //     const blackAndWhiteFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
    //     const resized_blackAndWhiteFrame = blackAndWhiteFrame.resize(100,100)
    //     const outputData = cv.imencode('.jpg', blackAndWhiteFrame).toString('base64');

    //     const final = `data:image/jpeg;base64,${outputData}`
        

      socket.emit('output', '');
  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});