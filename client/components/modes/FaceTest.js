import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import * as tf from "@tensorflow/tfjs";


const FaceTest = () => {

  const webcamRef = useRef(null);
  const cocoSsd = require("@tensorflow-models/coco-ssd");
  const [model, setModel] = useState(undefined);
  const [isLoaded, setIsLoaded] = useState(false);
  const [webcamEnabled, setWebcamEnabled] = useState(false);



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
      const obj = await net.detect(video);
    console.log(obj);
      // Draw mesh
 
    }
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
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

    </div>
  );
};

export default FaceTest;
