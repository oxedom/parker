import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
// import * as tf from "@tensorflow/tfjs";
import {
  renderRoi,
  renderAllOverlaps,
  clearCanvas,
} from "../libs/canvas_utility";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageWidthState, imageHeightState, selectedRoiState } from "./states";

const ClientRender = ({ processing, showDetections }) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const webcamRef = useRef(null);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);

  // const overlayEl = useRef(null);
  let overlayXRef = useRef(null);

  //Uncomment this if you don't want to the user to load tensorflow from Google API
  //And comment out import
  // const cocoSsd = require("@tensorflow-models/coco-ssd");

  useEffect(() => {
    // Need to do this for canvas2d to work
    const overlayEl = overlayXRef.current;
    if (overlayEl != null) {
      overlayXRef.current = overlayEl.getContext("2d");
    }
  }, []);

  const detect = async (net) => {
    // console.log(processing);
    // Check data is available
    if (!processing) {
      return;
    }
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

      let predictionsArr = [
        {
          cords: {
            right_x:-999,
            top_y: -999,
            width: -999,
            height: -999,
          },

          label: "Nothing",
          confidenceLevel: 99,
          area: -999,
        },
      ];

      for (let n = 0; n < predictions.length; n++) {
        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) {
          const right_x = predictions[n].bbox[0];
          const top_y = predictions[n].bbox[1];
          const width = predictions[n].bbox[2];
          const height = predictions[n].bbox[3];
          const label = predictions[n].class;

          const confidenceLevel = predictions[n].score;
          const obj = {
            cords: {
              right_x,
              top_y,
              width,
              height,
            },

            label,
            confidenceLevel,
            area: Math.ceil(width * height),
          };
          predictionsArr.push(obj);
        }
      }

      let action = {
        event: "occupation",
        payload: { predictionsArr: predictionsArr },
      };
      //Sends action request with a payload, the event is handled
      //inside the state event.
      setSelectedRois(action);
      if (showDetections) {
        
        renderAllOverlaps(predictionsArr, overlayXRef, imageWidth, imageHeight);
      }
    }
  };

  const runCoco = async () => {
    let id;
    const net = await cocoSsd.load();

    id = setInterval(() => {
      detect(net);
    }, 10);
    return id;
  };

  useEffect(() => {
    let intervalID;
    if (processing) {
      runCoco().then((id) => {
        intervalID = id;
      });
    }

    return function () {
      clearCanvas(overlayXRef, imageWidth, imageHeight);
      clearInterval(intervalID);
    };
  }, [processing, showDetections]);

  return (
    <>
      <canvas
        id="overlap-overlay"
        ref={overlayXRef}
        width={imageWidth}
        height={imageHeight}
        className="fixed"
      ></canvas>

      <Webcam
        height={imageHeight}
        width={imageWidth}
        style={{ height: imageHeight }}
        videoConstraints={{ height: imageHeight, video: imageWidth }}
        ref={webcamRef}
        muted={true}
        className=""
      />
    </>
  );
};

export default ClientRender;
