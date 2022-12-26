


import React, { useEffect, useRef, useState} from "react";
import io from 'socket.io-client';

const socket = io('http://localhost:2000/')

const Camera = () => {


  function isValidBase64Image(str) {
    const regex = /^data:image\/(png|jpeg|jpg|gif|bmp|webp);base64,[A-Za-z0-9+/]+=*$/;
    return regex.test(str);
  }
  

  const updateOutput = (outputFrame) => {


    let output = outputRef.current;
    output.src = outputFrame;

    window.requestAnimationFrame(updateOutput);
  }



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



        setInterval(async () => {

          const track = stream.getVideoTracks()[0]
          let imageCapture = new ImageCapture(track)
          const capturedImage = await imageCapture.takePhoto()
          const imageBuffer = await capturedImage.arrayBuffer()
          // const image_text = await capturedImage.text()
          
          const fd = new FormData()
          fd.append('blob', capturedImage)
          socket.emit('stream', capturedImage);  
          

          //Handle what node gives back
          socket.on('output', (data) => { 

        
            
      
          })
          
          
          // })

          // console.log(stream.getVideoTracks());
      //Sets the output base 64 Images to videooutput
 

     // get a single frame from the video stream
      // const frame = getFrame(video);

      // send the frame over the socket connection
      // socket.emit('stream', frame);
        //1 Frame per secound
        }, 1000)


      })
      .catch(err => {
        console.error("error:", err);
      });
  };

    useEffect(() => {

    getVideo();
  }, [inputRef]);






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