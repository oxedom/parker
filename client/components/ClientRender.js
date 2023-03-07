import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
import { clearCanvas } from "../libs/canvas_utility";
import EnableWebcam from "./EnableWebcam";
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
import { detectWebcam, getSetting } from "../libs/utillity";

const ClientRender = ({
  demo,
  processing,
  setLoadedCoco,
  loadedCoco,
  allowWebcam,
}) => {
  const model_dim = [640, 640];
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const fps = useRecoilValue(fpsState);
  const detectionThreshold = useRecoilValue(detectionThresholdState);
  const thresholdIou = useRecoilValue(thresholdIouState);
  let model;
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);
  let overlayXRef = useRef(null);
  const webcamRef = useRef(null);
  const demoRef = useRef(null);
  const [demoLoaded, setDemoLoaded] = useState(true);
  const [webcamLoaded, setWebcamLoaded] = useState(false);
  const modelName = "yolov7";
  const [ webcamPlaying , setWebcamPlaying] = useState(false)

  const handleDemoLoaded = (e) => 
  {
    setImageWidth(e.target.videoWidth);
    setImageHeight(e.target.videoHeight);
    setDemoLoaded(true)
    setAutoDetect(true)
  }

  async function setUserSettings() {
    let { width, height } = await getSetting();
    console.log(width, height);
    setImageWidth(width);
    setImageHeight(height);
  }

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

  useEffect(() => {
    let interval_load_webcam_id;
    if (allowWebcam && !webcamLoaded) {
      interval_load_webcam_id = setInterval(() => {
        detectWebcam(async (hasWebcamBoolean) => {
          if (hasWebcamBoolean) {
            await setUserSettings().then(() => {
              setWebcamLoaded(true);
              clearInterval(interval_load_webcam_id);
            });
          }
        });
      }, 1000);
    }
    return () => {
      clearInterval(interval_load_webcam_id);
    };
  }, [allowWebcam]);

  useEffect(() => {
    // Need to do this for canvas2d to work
    if (overlayXRef.current != null && loadedCoco) {
      overlayXRef.current = overlayXRef.current.getContext("2d");
    }
  }, [loadedCoco]);

  const detectFrame = async (model) => {
    if (!webcamRunning && !demo && webcamPlaying)  {
      return false;
    }

    let video;
    let videoWidth;
    let videoHeight;

    if (!demo && webcamRef.current != null) {
   
      video = webcamRef.current.video;
      videoWidth = webcamRef.current.video.videoWidth;
      videoHeight = webcamRef.current.video.videoHeight;
      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
    } else if (demo && demoLoaded && demoRef.current != null) {
      video = demoRef.current;
      videoWidth = demoRef.current.videoWidth;
      videoHeight = demoRef.current.videoHeight;
    } else {
      return;
    }

    tf.engine().startScope();

    let input = tf.tidy(() => {
      return tf.image
        .resizeBilinear(tf.browser.fromPixels(video), model_dim)
        .div(255.0)
        .transpose([2, 0, 1])
        .expandDims(0);
    });

    let res = model.execute(input);

    res = res.arraySync()[0];
    //Filtering only detections > conf_thres
    res = res.filter((dataRow) => dataRow[4] >= detectionThreshold);

    let boxes = [];
    let class_detect = [];
    let scores = [];

    res.forEach(process_pred);

    function process_pred(res) {
      let box = res.slice(0, 4);

      const cls_detections = res.slice(5, 85);
      let max_score_index = cls_detections.reduce(
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
        if (isVehicle(dect_label)) {
          const roiObj = { cords: {} };
          let [x1, y1, x2, y2] = xywh2xyxy(boxes[detectionIndex]);

          // Extract the bounding box coordinates from the 'boxes' tensor
          y1 = y1 * (imageHeight / 640);
          y2 = y2 * (imageHeight / 640);
          x1 = x1 * (imageWidth / 640);
          x2 = x2 * (imageWidth / 640);

          let dect_width = x2 - x1;
          let dect_weight = y2 - y1;
          roiObj.cords.bottom_y = y2;
          roiObj.cords.left_x = x2;
          roiObj.cords.top_y = y1;
          roiObj.cords.right_x = x1;
          roiObj.cords.width = dect_width;
          roiObj.cords.height = dect_weight;
          // Add the detection score to the bbox object
          roiObj.confidenceLevel = detectionScore;
          roiObj.label = dect_label;
          roiObj.area = dect_width * dect_weight;
          // Add the bbox object to the bboxes array

          predictionsArr.push(roiObj);
        }
      }
    }
    //Sends action request with a payload, the event is handled
    //inside the state event.
    let action = {
      event: "occupation",
      payload: { predictionsArr: predictionsArr, canvas: overlayXRef },
    };

    Promise.all([tf.dispose(input), tf.dispose(res)]).catch((err) => {
      console.error("Memory leak in coming");
      console.error(err);
    });

    console.log(tf.memory());

    tf.engine().endScope();
    setSelectedRois(action);
  };

  const runYolo = async () => {
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
    model = yolov7;
    console.log(model);
    id = setInterval(() => {
      if (true || !demo) {
        detectFrame(yolov7);
      }
    }, fps * 1000);

    return id;
  };

  useEffect(() => {
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
      setLoadedCoco(false);
      tf.dispose(model);
      // setModel(undefined);
    };
  }, [processing, imageHeight, imageWidth]);

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


      {!demo && webcamLoaded ? (
        <Webcam
          height={imageHeight}
          width={imageWidth}
          onPlay={()=> { setWebcamPlaying(true )}}
          style={{ height: imageHeight }}
          videoConstraints={{ height: imageHeight, video: imageWidth }}
          ref={webcamRef}
          muted={true}
          className=""
        />
      ) : null}

      {!demo && !webcamLoaded ? 
       <video
       height={imageHeight}
       width={imageWidth}
       >

      </video> : ""}


      {demo ? (
        <video
          ref={demoRef}
          muted={true}
          width={imageWidth}
          height={imageHeight}
          onLoad={(e) => {}}
          loop={true}
          onPlay={(e) => {

            handleDemoLoaded(e)
          }}
          autoPlay
          type="video/mp4"
          src="./demo_encode.mp4"
        />
      ) : null}
    </>
  );
};

export default ClientRender;
