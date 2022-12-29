import React, { useEffect, useRef, useState } from "react";
import io from "socket.io-client";
import axios from 'axios';

const flask_url = "http://localhost:5000/cv3";

const Camera = () => {
  const updateOutput = (outputFrame) => {
    let output = outputRef.current;
    output.src = outputFrame;

    window.requestAnimationFrame(updateOutput);
  };

  const inputRef = useRef(null);
  const outputRef = useRef(null);

  const getFrame = (videoStream) => {
    // create a canvas element to draw the video frame on
    const canvas = document.createElement("canvas");
    // get the canvas context
    const ctx = canvas.getContext("2d");
    // set the canvas width and height to match the video dimensions
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;

    // draw the current video frame onto the canvas
    ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    // get the image data from the canvas

    const imageData = canvas.toDataURL("image/jpeg");

    return imageData;
  };

  const getVideo = () => {
    

    navigator.mediaDevices
      .getUserMedia({ video: { width: 720 } })
      .then((stream) => {
        //Gets the current screen
        let video = inputRef.current;
        //Sets the Src of video Object
        video.srcObject = stream;
        //Plays the video
        video.play();

        setInterval(async () => {
          const track = stream.getVideoTracks()[0];
          let imageCapture = new ImageCapture(track);
          const capturedImage = await imageCapture.takePhoto();
          const imageBuffer = await capturedImage.arrayBuffer();

          const res = await axios.post(flask_url, { buffer: imageBuffer})
          const base64 = `data:image/jpg;base64,${res.data}`
          updateOutput(base64)

          //Handle what node gives back
      
        }, 10000);
      })
      .catch((err) => {
        console.error("error:", err);
      });
  };


  //   useEffect(() => {

  //   getVideo();
  // }, [inputRef]);

  getVideo();

  return (
    <div>
      <div>
        <h1> Client </h1>

        <video ref={inputRef} />
        <h1> Server </h1>
        <img alt="sever" ref={outputRef} />
      </div>
    </div>
  );
};

export default Camera;
