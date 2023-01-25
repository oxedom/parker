import { useState, useEffect, useRef, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from 'react-webcam';

const cocoSsd = require("@tensorflow-models/coco-ssd");

const ClientRender = () => {

  const videoRef = useRef(null)
  const webcamRef = useRef(null);


  const [isLoadded, setLoaded] = useState(false)
  const  [webcamEnabled, setWebcamEnabled] = useState(false)


  const loadModels = async () => {
    const modelUrl = '/models';
    await Promise.all([,
      cocoSsd.load()
    ]);

  }

    const handleLoadWaiting = async () => {
      return new Promise((resolve) => {
        const timer = setInterval(() => {
          const conditions = (
            webcamRef.current &&
            webcamRef.current.video &&
            webcamRef.current.video.readyState === 4
          );
  
          if (conditions) {
            console.log("Done loading waiting");
            resolve(true);
            clearInterval(timer);
          }
        }, 500);
      });
    };


    const objectDecterHandler = async () => 
    {
      console.log("Object decter handler");
      await loadModels();
      await handleLoadWaiting();
      console.log(webcamRef.current.video);
      if(webcamRef.current.video) 
      {
        setLoaded(true)
        console.log("Webcam loaded");
      }
 
    }


  useEffect(() => {
 
   if(webcamEnabled) 
   {
    setLoaded(false)
    objectDecterHandler()
   }
  }, [webcamEnabled]);

  function predictWebcam() {
    // Now let's start classifying the stream.
    model.detect(videoRef.current).then(function (predictions) {
      // Remove any highlighting we did previous frame.

      // Now lets loop through predictions and draw them to the live view if
      // they have a high confidence score.
      for (let n = 0; n < predictions.length; n++) {
        console.log(predictions);
        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) {
          const p = document.createElement("p");
          p.innerText =
            predictions[n].class +
            " - with " +
            Math.round(parseFloat(predictions[n].score) * 100) +
            "% confidence.";
          // Draw in top left of bounding box outline.
          p.style =
            "left: " +
            predictions[n].bbox[0] +
            "px;" +
            "top: " +
            predictions[n].bbox[1] +
            "px;" +
            "width: " +
            (predictions[n].bbox[2] - 10) +
            "px;";

          // Draw the actual bounding box.
          const highlighter = document.createElement("div");
          highlighter.setAttribute("class", "highlighter");
          const left_x = predictions[n].bbox[0];
          const top_y = predictions[n].bbox[1];
          const width = predictions[n].bbox[2];
          const height = predictions[n].bbox[3];
          console.log(predictions);
          const obj = { left_x, top_y, width, height };
          console.log(obj);

     
        }
      }

      // Call this function again to keep predicting when the browser is ready.
      window.requestAnimationFrame(predictWebcam);
    });
  }

  
  return (
    <div>
   <input type='checkbox' checked={webcamEnabled} onClick={() => setWebcamEnabled((previous) => !previous)} />
      <Webcam ref={webcamRef} ></Webcam>
    </div>
  );
};

export default ClientRender;
