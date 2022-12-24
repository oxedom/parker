
import React from 'react';
import  Webcam from 'react-webcam';
import io from 'socket.io-client';


const socket = io.connect('http://localhost:2000');


socket.on('message', msg => console.log(msg));


const Camera = () => {
  return (
    <div>
    
      <h1>I love Lihi {'<3'} </h1> 
      <Webcam onUserMedia={(stream) => { io('')}}audio={false} video={true} height={480} width={640} />
    </div>
  );
};

export default Camera;