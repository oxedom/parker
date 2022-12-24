
import React, { useState, useEffect } from 'react';
import socketio from 'socket.io-client';

const Webcam = () => {
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const [socket, setSocket] = useState(null);

  const handleClick = () => {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(stream => {
        setStream(stream);
      })
      .catch(error => {
        setError(error);
      });
  }

  useEffect(() => {
    if (stream) {
      console.log(stream);
      const socket = socketio.connect('http://localhost:2000');

      socket.on('connect', () => {
        console.log('Connected to the server');
      });

      const sendData = () => {
        socket.emit('stream', stream);
      };

      const intervalId = setInterval(sendData, 1000); // send data every second

      setSocket(socket);

      return () => {
        clearInterval(intervalId);
        socket.disconnect();
      };
    }
  }, [stream]);

  return (
    <div>
      {error && <p>{error.message}</p>}
      <button onClick={handleClick}>Access Webcam</button>
    </div>
  );
};

export default Webcam;