


import React, { useEffect, useRef, useState} from "react";
import io from 'socket.io-client';

const socket = io('http://localhost:5000/')

const Camera = () => {

  const [videoOutput, setVideoOutput] = useState()

  const inputRef = useRef(null);
  const outputRef = useRef(null)


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


  const getVideo = () => {
    socket.connect()
    navigator.mediaDevices
      .getUserMedia({ video: { width:  720} })
      .then(stream => {
        
        //Gets the current screen
        let video = inputRef.current;
        //Sets the Src of video Object
        video.srcObject = stream;
        //Plays the video
        video.play();

        // setInterval(() => {
      
      //Sets the output base 64 Images to videooutput
      socket.on('output', (data) => { setVideoOutput(data)}) 
      socket.on('matt', (data) => { console.log(data);}) 
        
     // get a single frame from the video stream
      const frame = getFrame(video);
      // send the frame over the socket connection
      if(videoOutput !== null) {socket.emit('stream', frame);}

    

        //23 Frame per secound
        // }, 10)


      })
      .catch(err => {
        console.error("error:", err);
      });
  };

    useEffect(() => {
    getVideo();
  }, [inputRef]);






  useEffect(() => {
    const updateOutput = () => {

      let output = outputRef.current;
      if(videoOutput != null) {  output.src = videoOutput;}
  
      window.requestAnimationFrame(updateOutput);
    }


    updateOutput();
  }, [videoOutput]);

  return (
    <div>
      <div>
        <h1> Client </h1>

        <video ref={inputRef} />
        <h1> Server </h1>
        <img alt='sever' ref={outputRef} /> 
      </div>
    </div>
  );

};

export default Camera;