import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import * as tf from "@tensorflow/tfjs";


const FaceTest = () => {

  const webcamRef = useRef(null);
  const cocoSsd = require("@tensorflow-models/coco-ssd");


  const detect = async (net) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

  

      // Make Detections
      const predictions = await net.detect(video);

      for (let n = 0; n < predictions.length; n++) {

        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) { 
          const right_x = predictions[n].bbox[0];
          const top_y = predictions[n].bbox[1];
          const width = predictions[n].bbox[2];
          const height = predictions[n].bbox[3];
          const label = predictions[n].class
          const confidenceLevel = predictions[n].score
          const obj = { right_x, top_y, width, height, label, confidenceLevel };
           
        }



 
    } }
  };




const runCoco = async () => 
{
  const net = await cocoSsd.load()

  setInterval(() => {

    detect(net)
  }, 50);

}

useEffect(() => { runCoco()}, [])

  return (
    <div className="">
              <Webcam
          ref={webcamRef}
          muted={true} 
          className=""
        />
    </div>

  );
};

export default FaceTest;
