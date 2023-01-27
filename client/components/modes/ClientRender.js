import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import * as tf from "@tensorflow/tfjs";
import {  renderRoi } from "../../libs/canvas_utility"
import { useRecoilValue, useRecoilState } from "recoil";
import {
  imageWidthState,
  imageHeightState,
} from "../../components/states";

const ClientRender = ({setLoaded, setTotalFrames, webcamApproved, loaded}) => {

  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);


  const webcamRef = useRef(null);
  // const overlayEl = useRef(null);
  let overlayXRef = useRef(null);
  const cocoSsd = require("@tensorflow-models/coco-ssd");

  function renderAllOverlaps(overlaps) {
    //Clears canvas before rendering all overlays (Runs each response)
    //For each on the detections
    overlaps.forEach((o) => {
      
      o.color = "#FFFF00"
      console.table(o.cords);
      renderRoi(o, overlayXRef, "#FFFF00");
    });



  }

  useEffect(() => {
    //Need to do this for canvas2d to work
 
    const overlayEl = overlayXRef.current;
    overlayXRef.current = overlayEl.getContext("2d")

  }, []);



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
      let arr = []
      for (let n = 0; n < predictions.length; n++) {

        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) { 
          
          const right_x = predictions[n].bbox[0];
          const top_y = predictions[n].bbox[1];
          const width = predictions[n].bbox[2];
          const height = predictions[n].bbox[3];
          const label = predictions[n].class
          const confidenceLevel = predictions[n].score
          const obj = { 
            cords: {
              right_x, top_y, width, height,
            },
          
           label, confidenceLevel };
          arr.push(obj)
           
        }
        renderAllOverlaps(arr)
        setTotalFrames((prev => { return prev+1}))
        


 
    } }
  };




const runCoco = async () => 
{
  const net = await cocoSsd.load()
  setLoaded(true)
  setInterval(() => {
    
    detect(net)
  }, 50);

}

useEffect(() => { runCoco()}, [])

  return (

      <>
                <canvas
        id="overlap-overlay"
        ref={overlayXRef}
        width={imageWidth}
        height={imageHeight}
        className="fixed"
      ></canvas>

    {(!webcamApproved && !loaded) && (<h1> Webcan not apoprved and need to load model</h1>)}
    {(webcamApproved && !loaded) && <h1> Still loading model Shit </h1>}
    {webcamApproved && loaded && (<div className="">
              <Webcam
          ref={webcamRef}
          muted={true} 
          className=""
        />
    </div>) }
      
      
      
      </>


  );
};

export default ClientRender;
