import React, { useEffect, useRef, useState } from "react";
import io from 'socket.io-client';


const socket = io('http://localhost:2000/')

const Camera = () => {

  const [videoOutput, setVideoOutput] = useState()
  const getFrame = (videoStream) => {
    // create a canvas element to draw the video frame on
    const canvas = document.createElement('canvas');
    // get the canvas context
    const ctx = canvas.getContext('2d');
    // set the canvas width and height to match the video dimensions
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
  
    // draw the current video frame onto the canvas
    ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    // get the image data from the canvas
    const imageData = canvas.toDataURL('image/jpeg');
  
    return imageData;
  };

  const videoRef = useRef(null);
  const getVideo = () => {
    socket.connect()
    navigator.mediaDevices
      .getUserMedia({ video: { width:  720} })
      .then(stream => {
        
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
        // socket.emit('stream', stream);

        


        setInterval(() => {
          // draw the current video frame onto the canvas
       
          // get the image data from the canvas
          const result = getFrame(video)
     
          // send the image data over the socket connection as a binary string
          socket.emit('stream', result );
        }, 1000 / 30); // send 30 frames per second


      })
      .catch(err => {
        console.error("error:", err);
      });
    socket.on('output', (data) => {setVideoOutput(data)}) 
  };

    useEffect(() => {
    getVideo();
  }, [videoRef]);



  return (
    <div>
      <div>
  
        <video ref={videoRef} />
        <h1> Server </h1>
        <image ref={videoOutput}> </image>
      </div>
    </div>
  );

};

export default Camera;