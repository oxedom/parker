import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import * as tf from "@tensorflow/tfjs";
import Loader from "./Loader";
import { clearCanvas } from "../libs/canvas_utility";

import { useRecoilValue, useRecoilState } from "recoil";
import {
  imageWidthState,
  imageHeightState,
  selectedRoiState,
  thresholdIouState,
  detectionThresholdState,
  fpsState,
  vehicleOnlyState,
  autoDetectState,
} from "./states";
import {
  processInputImage,
  processDetectionResults,
  nmsDetectionProcess,
} from "../libs/tensorflow_utility";
import {
  detectWebcam,
  getSetting,
  webcamRunning,
  detectionsToROIArr,
} from "../libs/utillity";
import LoadingScreen from "./LoadingScreen";

const ClientRender = ({
  demo,
  processing,
  WebRTCMode,
  setLoadedCoco,
  loadedCoco,
  setDemoLoaded,
  webcamPlaying,
  setWebRTCMode,
  setAllowWebcam,
  setWebcamPlaying,
  demoLoaded,
  WebRTCLoaded,
  setWebRTCLoaded,
  allowWebcam,
  rtcOutputRef,
}) => {
  const model_dim = [640, 640];
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const fps = useRecoilValue(fpsState);
  const detectionThreshold = useRecoilValue(detectionThresholdState);
  const thresholdIou = useRecoilValue(thresholdIouState);
  const vehicleOnly = useRecoilValue(vehicleOnlyState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);


  let overlayXRef = useRef(null);

  const webcamRef = useRef(null);
  const demoRef = useRef(null);
  const modelRef = useRef(null);
  const [webcamLoaded, setWebcamLoaded] = useState(false);
  const [loadingYolo, setYoloLoading] = useState({
    loaded: false,
    progress: 0,
  });
  const videoConstraints = {
    maxWidth: 1280,
    maxHeight: 720,
  };
  const modelName = "yolov7";

  const handleDemoLoaded = (e) => {
    setImageWidth(e.target.videoWidth);
    setImageHeight(e.target.videoHeight);
    setDemoLoaded(true);
    setWebcamPlaying(false);
    setAutoDetect(true);
  };

  async function setUserSettings() {
    let { width, height } = await getSetting();

    setImageWidth(width);
    setImageHeight(height);
  }

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
    if (!webcamRunning && !demo && !webcamPlaying && !WebRTCLoaded) {
      return false;
    }

    let video;
    let videoWidth;
    let videoHeight;
    if (rtcOutputRef.current != null && WebRTCLoaded) {
      video = rtcOutputRef.current;
      videoWidth = rtcOutputRef.current.videoWidth;
      videoHeight = rtcOutputRef.current.videoHeight;
    } else if (!demo && webcamRef.current != null) {
      video = webcamRef.current.video;
      videoWidth = webcamRef.current.video.videoWidth;
      videoHeight = webcamRef.current.video.videoHeight;
      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      //If demo
    } else if (demo && demoLoaded && demoRef.current != null) {
      video = demoRef.current;

      videoWidth = demoRef.current.videoWidth;
      videoHeight = demoRef.current.videoHeight;
    } else {
      return;
    }

    tf.engine().startScope();

    let input = processInputImage(video, model_dim);

    let res = model.execute(input);

    const { boxes, class_detect, scores } = processDetectionResults(
      res,
      detectionThreshold
    );

    const { detectionIndices } = await nmsDetectionProcess(
      boxes,
      scores,
      thresholdIou
    );

    const predictionsArr = detectionsToROIArr(
      detectionIndices,
      boxes,
      class_detect,
      scores,
      imageWidth,
      imageHeight,
      vehicleOnly,
      boxes,
      class_detect,
      scores
    );

    //Sends action request with a payload, the event is handled
    //inside the state event.
    let action = {
      event: "occupation",
      payload: { predictionsArr: predictionsArr, canvas: overlayXRef },
    };

    //Disposing tensors
    Promise.all([tf.dispose(input), tf.dispose(res)]).catch((err) => {
      console.error("Memory leak in coming");
      console.error(err);
    });

    tf.engine().endScope();
    clearCanvas(overlayXRef, imageWidth, imageHeight);
    setSelectedRois(action);
  };

  const runYolo = async () => {
    let id;
    if (modelRef.current === null) {
      let yolov7 = await tf.loadGraphModel(
        `${window.location.origin}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            //Loading
            setYoloLoading({ loaded: false, progress: fractions });
          },
        }
      );

      setYoloLoading({ loaded: true, progress: 0 });
      modelRef.current = yolov7;
      const dummyInput = tf.ones(yolov7.inputs[0].shape);
      const warmupResult = await yolov7.executeAsync(dummyInput);
      tf.dispose(warmupResult);
      tf.dispose(dummyInput);
      setLoadedCoco(true);

      id = setInterval(() => {
        detectFrame(yolov7);
      }, fps * 1000);
    } else {
      id = setInterval(() => {
        detectFrame(modelRef.current);
      }, fps * 1000);
    }

    return id;
  };

  useEffect(() => {
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

      if (modelRef.current) {
        // modelRef.current.dispose();
      }

      // setLoadedCoco(false);
    };
  }, [processing, imageHeight, imageWidth, WebRTCLoaded]);

  return loadingYolo.loaded ? (
    <>
      {loadedCoco  ? (
        <canvas
          id="overlap-overlay"
          ref={overlayXRef}
          width={imageWidth}
          height={imageHeight}
          className="fixed "
        ></canvas>
      ) : null}

      {!demo && !webcamLoaded && WebRTCMode ? (
        <video
          id="webRTC"
          ref={rtcOutputRef}
          muted={true}
          onPlay={(e) => {
            setWebRTCLoaded(true);
          }}
          autoPlay={true}
          width={imageWidth}
          height={imageHeight}
        ></video>
      ) : null}

      {/* Webcam */}
      {!demo && webcamLoaded ? (
        <Webcam
          height={imageHeight}
          width={imageWidth}
          onPlay={() => {
            setDemoLoaded(false);
            setWebcamPlaying(true);
          }}
          // style={{ height: imageHeight }}
          videoConstraints={videoConstraints}
          ref={webcamRef}
          muted={true}
          className=""
        />
      ) : null}

      {!demo && !webcamLoaded && !WebRTCMode ? (
        <LoadingScreen
          setAllowWebcam={setAllowWebcam}
          setWebRTCMode={setWebRTCMode}
        />
      ) : (
        ""
      )}

      {demo ? (
        <video
          ref={demoRef}
          muted={true}
          width={imageWidth}
          height={imageHeight}
          onLoad={(e) => {}}
          loop={true}
          onPlay={(e) => {
            handleDemoLoaded(e);
          }}
          autoPlay
          type="video/mp4"
          src="./demo.mp4"
        />
      ) : null}
    </>
  ) : (
    <Loader progress={loadingYolo.progress} />
  );
};

export default ClientRender;
