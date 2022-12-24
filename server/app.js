const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);
const cv = require('opencv4nodejs');
const cors = require('cors');
const { log } = require('console');

app.use(cors());
app.use(() => { console.log('big');})

io.on('connection', socket => {
  console.log('Connected to client');

  socket.on('stream', data => {
    console.log('Received webcam data from client');

    // Convert the data to a matrix
    const frame = cv.imdecode(Buffer.from(data, 'base64'));

    // Convert the colors to black and white
    const blackAndWhiteFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);

    // Convert the matrix back to data that can be emitted to the client
    const outputData = cv.imencode('.jpg', blackAndWhiteFrame).toString('base64');

    // Emit the processed data to the client
    socket.emit('output', outputData);
  });
});

server.listen(2000, () => {
  console.log('Server listening on port 2000');
});