import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
// import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
import {
  renderRoi,
  renderAllOverlaps,
  clearCanvas,
} from "../libs/canvas_utility";
import { renderBoxes, xywh2xyxy } from "../utils/renderBox.js";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageWidthState, imageHeightState, selectedRoiState } from "./states";

const ClientRender = ({ processing, showDetections }) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const webcamRef = useRef(null);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);
  const [loadedCoco, setLoadedCoco] = useState(false);
  // const overlayEl = useRef(null);
  let overlayXRef = useRef(null);
  let loadingRef = useRef(null);
  const modelName = "yolov7";
  const threshold = 0.8;

  //Uncomment this if you don't want to the user to load tensorflow from Google API
  //And comment out import
  // const cocoSsd = require("@tensorflow-models/coco-ssd");

  useEffect(() => {
    // Need to do this for canvas2d to work
    if (loadedCoco) {
      const overlayEl = overlayXRef.current;
      if (overlayEl != null) {
        overlayXRef.current = overlayEl.getContext("2d");
      }
    } else {
      // const loadingEL = loadingRef.current
      // if (loadingEL != null) {
      //   loadingRef.current = loadingEL.getContext("2d")
      //   var loadingText = "Loading...";
      //   loadingRef.current.fillStyle = "green";
      //   loadingRef.current.fillRect(0, 0, imageWidth, imageHeight)
      //   loadingRef.current.font = "48px sans-serif";
      //   loadingRef.current.fillText(loadingText, imageWidth / 2, imageHeight / 2);
      // }
    }
  }, [loadedCoco]);

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
            right_x: -999,
            top_y: -999,
            width: -999,
            height: -999,
          },

          label: "EMPTY_ROI",
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

  const detectFrame = async (model) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      let start = Date.now()
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      
      const model_dim = [640, 640];
      tf.engine().startScope();

      const input = tf.tidy(() => {
        const img = tf.image
          .resizeBilinear(tf.browser.fromPixels(video), model_dim)
          .div(255.0)
          .transpose([2, 0, 1])
          .expandDims(0);

        return img;
      });

      let res = model.execute(input) 

      // await model.execute(input).then((res) => {
        res = res.arraySync()[0];
        //Filtering only detections > conf_thres
        const conf_thres = 0.25;
        res = res.filter((dataRow) => dataRow[4] >= conf_thres);

        var boxes = [];
        var class_detect = [];
        var scores = [];
        var roiObjs = [];

        res.forEach(process_pred);

        function process_pred(res) {
          var box = res.slice(0, 4);

          // //non_max_suppression

          const roiObj = {
            cords: {},
          };
          const cls_detections = res.slice(5, 85);
          var max_score_index = cls_detections.reduce(
            (imax, x, i, arr) => (x > arr[imax] ? i : imax),
            0
          );
          const search_index = class_detect.indexOf(max_score_index);
          roiObj.score = max_score_index;
      
          roiObj.label = labels[max_score_index];

          boxes.push(box);
          scores.push(res[max_score_index + 5]);
          class_detect.push(max_score_index);
          
          tf.dispose(res);
        }



        if(boxes.length === 0) { return}
        const iouThreshold = 0.1;
          const scoreThreshold = 0;
          const maxOutputSize = 20;
          let indices =  await tf.image.nonMaxSuppressionAsync(
            boxes,
            scores,
            maxOutputSize,
            iouThreshold,
            scoreThreshold
          );

          // Keep only the indices with a high enough score
          indices = indices.dataSync();
          let filteredBoxes = [];
          let filteredScores = [];
          let filteredClasses = [];
          
          for (let i = 0; i < indices.length; i++) {
            // boxes[indices[i]][0] = boxes[indices[i]][0]*[imageWidth/640]
            // boxes[indices[i]][1]= boxes[indices[i]][1]*[imageHeight/640]
            // boxes[indices[i]][2]= boxes[indices[i]][2]*[imageWidth/640]
            // boxes[indices[i]][3]= boxes[indices[i]][1]*[imageHeight/640]
            filteredBoxes.push(boxes[indices[i]]);
            filteredScores.push(scores[indices[i]]);
            filteredClasses.push(class_detect[indices[i]]);

            
          }
        
        
          // let [x1, y1, x2, y2] = xywh2xyxy(filteredBoxes[0]);
          // const width = x2 - x1;
          // const height = y2 - y1;

          // if (roiObjs.length === 0) {
          //   roiObj.cords.right_x = x1;
          //   roiObj.cords.top_y = y1;
          //   roiObj.cords.width = width;
          //   roiObj.cords.height = height;
          //   roiObjs.push(roiObj);
          // }
        
        if(showDetections && filteredBoxes.length > 0) 
        {
          renderBoxes(
            overlayXRef.current,
            threshold,
            filteredBoxes,
            filteredScores,
            filteredClasses
          );
    
        }
      

      
      // }
      
      
  
    
      tf.engine().endScope();
      let end = Date.now()
      console.log(end-start);
    }

  };

  const runYolo = async () => 
  {
    let id;
    let yolov7 = await tf.loadGraphModel(
      `${window.location.origin}/${modelName}_web_model/model.json`,
      {
        onProgress: (fractions) => {},
      }
    )
    setLoadedCoco(true);
    const dummyInput = tf.ones(yolov7.inputs[0].shape);
    const warmupResult = await yolov7.executeAsync(dummyInput)
    tf.dispose(warmupResult);
    tf.dispose(dummyInput);
    id = setInterval(() => {
      detectFrame(yolov7) // get another frame
    }, 200);
    return id; 
  }

  useEffect(() => {
    let intervalID;
    if (processing) {
      runYolo().then((id) => {
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
      {loadedCoco ? (
        <canvas
          id="overlap-overlay"
          ref={overlayXRef}
          width={imageWidth}
          height={imageHeight}
          className="fixed"
        ></canvas>
      ) : null}
      {loadedCoco ? (
        <Webcam
          height={imageHeight}
          width={imageWidth}
          style={{ height: imageHeight }}
          videoConstraints={{ height: imageHeight, video: imageWidth }}
          ref={webcamRef}
          muted={true}
          className=""
        />
      ) : (
        <canvas
          ref={loadingRef}
          height={imageHeight}
          width={imageWidth}
        ></canvas>
      )}
    </>
  );
};

export default ClientRender;
