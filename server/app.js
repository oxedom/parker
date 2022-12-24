const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server, {  cors: {
  origin: "http://localhost:3000",
  methods: ["GET", "POST"]
}});


const cv = require('opencv4nodejs');
const cors = require('cors');

app.use(cors());


  io.on('connection', socket => {

    socket.on('stream', data => {
  
      socket.emit('output', data);

  });


});



server.listen(2000, () => {
  console.log('Server listening on port 2000');
});