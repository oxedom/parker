

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
  

    

      try {
    
        const response = await axios.post('http://localhost:5000/cv3', { buffer: imageBuffer})
        console.log(response.data);
      } catch (error) {
        
      } 
   
      // python_socket.emit('buffer', decoded_buffer)
 
      
     
      socket.emit('output', '');
  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});