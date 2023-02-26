import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
// import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
import { renderAllOverlaps, clearCanvas } from "../libs/canvas_utility";
import { xywh2xyxy } from "../utils/renderBox.js";
import { useRecoilValue, useRecoilState } from "recoil";
import {
  imageWidthState,
  imageHeightState,
  selectedRoiState,
  thresholdIouState,
  detectionThresholdState,
  fpsState,
  autoDetectState,
} from "./states";
import { isVehicle } from "../libs/utillity";

const ClientRender = ({
  processing,

  showDetections,
  setProcessing,
  setLoadedCoco,
  loadedCoco,
}) => {
  const model_dim = [640, 640];
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const fps = useRecoilValue(fpsState);

  const [model, setModel] = useState(undefined);
  const detectionThreshold = useRecoilValue(detectionThresholdState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const thresholdIou = useRecoilValue(thresholdIouState);
  let overlayXRef = useRef(null);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);
  const webcamRef = useRef(null);
  const modelName = "yolov7";

  const webcamRunning = () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      return true;
    } else {
      return false;
    }
  };

  const createInput = (video) => {
    return tf.tidy(() => {
      const img = tf.image
        .resizeBilinear(tf.browser.fromPixels(video), model_dim)
        .div(255.0)
        .transpose([2, 0, 1])
        .expandDims(0);

      return img;
    });
  };

  useEffect(() => {
    // Need to do this for canvas2d to work

    if (overlayXRef.current != null && loadedCoco) {
      overlayXRef.current = overlayXRef.current.getContext("2d");
    }
  }, [loadedCoco]);

  const detectFrame = async (model) => {
    if (!webcamRunning) {
      return false;
    }

    const video = webcamRef.current.video;
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    // Set video width
    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;

    tf.engine().startScope();

    const input = createInput(video);

    let res = model.execute(input);

    res = res.arraySync()[0];
    //Filtering only detections > conf_thres
    res = res.filter((dataRow) => dataRow[4] >= detectionThreshold);

    let boxes = [];
    let class_detect = [];
    let scores = [];

    res.forEach(process_pred);

    function process_pred(res) {
      var box = res.slice(0, 4);

      const cls_detections = res.slice(5, 85);
      var max_score_index = cls_detections.reduce(
        (imax, x, i, arr) => (x > arr[imax] ? i : imax),
        0
      );

      boxes.push(box);
      scores.push(res[max_score_index + 5]);
      class_detect.push(max_score_index);
    }

    let nmsDetections;
    let detectionIndices;
    let detectionScores;
    let predictionsArr = [];
    if (boxes.length > 0) {
      nmsDetections = await tf.image.nonMaxSuppressionAsync(
        boxes,
        scores,
        100,
        thresholdIou
      );

      detectionIndices = nmsDetections.dataSync();

      for (let i = 0; i < detectionIndices.length; i++) {
        const detectionIndex = detectionIndices[i];
        const detectionScore = scores[detectionIndex];
        const detectionClass = class_detect[detectionIndex];
        let dect_label = labels[detectionClass];
        if (!isVehicle(dect_label)) {
          return;
        }
        const roiObj = { cords: {} };
        let [x1, y1, x2, y2] = xywh2xyxy(boxes[detectionIndex]);

        // Extract the bounding box coordinates from the 'boxes' tensor
        y1 = y1 * (imageHeight / 640);
        y2 = y2 * (imageHeight / 640);
        x1 = x1 * (imageWidth / 640);
        x2 = x2 * (imageWidth / 640);
        let width = x2 - x1;
        let height = y2 - y1;
        roiObj.cords.bottom_y = y2;
        roiObj.cords.left_x = x2;
        roiObj.cords.top_y = y1;
        roiObj.cords.right_x = x1;
        roiObj.cords.width = width;
        roiObj.cords.height = height;
        // Add the detection score to the bbox object
        roiObj.confidenceLevel = detectionScore;
        roiObj.label = dect_label;
        roiObj.area = width * height;
        // Add the bbox object to the bboxes array

        predictionsArr.push(roiObj);
      }
    }

    let action = {
      event: "occupation",
      payload: { predictionsArr: predictionsArr },
    };

    //Sends action request with a payload, the event is handled
    //inside the state event.
    setSelectedRois(action);

    if (showDetections && predictionsArr.length > 0) {
      renderAllOverlaps(predictionsArr, overlayXRef, imageWidth, imageHeight);
    } else {
      clearCanvas(overlayXRef, imageWidth, imageHeight);
    }

    tf.dispose(res);
    tf.engine().endScope();
  };

  const runYolo = async () => {
    console.log("LOAD YOLO HAS RUN ");
    let id;
    let yolov7 = await tf.loadGraphModel(
      `${window.location.origin}/${modelName}_web_model/model.json`,
      {
        onProgress: (fractions) => {},
      }
    );

    const dummyInput = tf.ones(yolov7.inputs[0].shape);
    const warmupResult = await yolov7.executeAsync(dummyInput);
    tf.dispose(warmupResult);
    tf.dispose(dummyInput);
    setLoadedCoco(true);
    setModel(yolov7);
    id = setInterval(() => {
      detectFrame(yolov7);
    }, fps * 1000);

    return id;
  };

  useEffect(() => {
    // console.log(" I HAVE FUN");
    console.log(
      "Load Yolo Use Effect Rerun",
      `Processing is currently: ${processing}`
    );
    let id;

    if (processing) {
      runYolo().then((res) => {
        id = res;
      });
    }

    //Clean up
    return function () {
      clearInterval(id);
      clearCanvas(overlayXRef, imageWidth, imageHeight);
      if (model != undefined) {
        tf.dispose(model);
        setLoadedCoco(false);
        setModel(undefined);
      }
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
      {
        <Webcam
          height={imageHeight}
          width={imageWidth}
          style={{ height: imageHeight }}
          videoConstraints={{ height: imageHeight, video: imageWidth }}
          ref={webcamRef}
          muted={true}
          className=""
        />
      }
    </>
  );
};

export default ClientRender;
