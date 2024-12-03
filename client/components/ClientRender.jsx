import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
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
  allowWebGPUState,
  vehicleOnlyState,
  autoDetectState,
} from "./states";
import {
  processInputImage,
  non_max_suppression,
} from "../libs/tensorflow_utility";
import { detectionsToROIArr } from "../libs/utillity";

import {
  detectWebcam,
  getSetting,
  webcamRunning,
} from "../libs/settings_utility";

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
  webcamLoaded,
  setWebcamLoaded,
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
  const allowWebGPU = useRecoilValue(allowWebGPUState)
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const [selectedRois, setSelectedRois] = useRecoilState(selectedRoiState);

  let overlayXRef = useRef(null);

  const webcamRef = useRef(null);
  const demoRef = useRef(null);
  const modelRef = useRef(null);

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
    if (selectedRois.length === 0) setAutoDetect(true);

    setImageWidth(e.target.videoWidth);
    setImageHeight(e.target.videoHeight);
    setDemoLoaded(true);
    setWebcamPlaying(false);
  };

  async function setUserSettings() {
    try {
      let { width, height } = await getSetting();

      setImageWidth(width);
      setImageHeight(height);
    } catch (error) {
      console.error(error);
    }
  }



  useEffect(() => {
    async function enableTFJSgpuBackend() {
      await import('@tensorflow/tfjs-backend-webgl')
    }

    async function enableTFJScpuBackend() {
      await import('@tensorflow/tfjs-backend-cpu')
    }


    if (allowWebGPU)
      enableTFJSgpuBackend()
    else enableTFJScpuBackend()

  }, [allowWebGPU])

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
      try {
        overlayXRef.current = overlayXRef.current.getContext("2d");
      } catch (error) {
        console.log(error);
      }
    }
  }, [loadedCoco]);

  function getVideoDims(type) {
    let video = null;
    let videoWidth = null;
    let videoHeight = null;

    if (type === "demo") {
      video = demoRef.current;
      videoWidth = demoRef.current.videoWidth;
      videoHeight = demoRef.current.videoHeight;
    } else if (type === "webcam") {
      video = webcamRef.current.video;
      videoWidth = webcamRef.current.video.videoWidth;
      videoHeight = webcamRef.current.video.videoHeight;
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
    } else if (type === "rtc") {
      video = rtcOutputRef.current;
      videoWidth = rtcOutputRef.current.videoWidth;
      videoHeight = rtcOutputRef.current.videoHeight;
    }
    return { video, videoHeight, videoWidth };
  }

  function getModeString() {
    if (rtcOutputRef.current != null && WebRTCLoaded) {
      return "rtc";
    } else if (!demo && webcamRef.current != null) {
      return "webcam";
    } else if (demo && demoLoaded && demoRef.current != null) {
      return "demo";
    } else {
      return null;
    }
  }

  const detectFrame = async (model) => {
    if (!webcamRunning && !demo && !webcamPlaying && !WebRTCLoaded) {
      return false;
    }

    let video = null;
    let mode = getModeString();
    if (mode === null) {
      return;
    }

    const dims = getVideoDims(mode);
    video = dims.video;

    tf.engine().startScope();

    let input = processInputImage(video, model_dim);

    let res = model.execute(input);

    const detections = non_max_suppression(
      res.arraySync()[0],
      detectionThreshold,
      thresholdIou,
      50
    );

    const predictionsRois = detectionsToROIArr(
      detections,
      imageWidth,
      imageHeight,
      vehicleOnly
    );

    //Sends action request with a payload, the event is handled
    //inside the state event.
    let action = {
      event: "occupation",
      payload: { predictionsArr: predictionsRois, canvas: overlayXRef },
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
    };
  }, [
    processing,
    imageHeight,
    imageWidth,
    WebRTCLoaded,
    demoLoaded,
    webcamLoaded,
  ]);

  return loadingYolo.loaded ? (
    <section className="overflow-hidden rounded-md">
      {loadedCoco ? (
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
          className={`${WebRTCLoaded ? "block" : "hidden"} `}
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
          videoConstraints={videoConstraints}
          ref={webcamRef}
          muted={true}
          className=""
        />
      ) : null}
      {!demo && !webcamLoaded && !WebRTCLoaded ? (
        <LoadingScreen
          setAllowWebcam={setAllowWebcam}
          setWebRTCMode={setWebRTCMode}
          allowWebcam={allowWebcam}
          WebRTCMode={WebRTCMode}
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
          loop={true}
          onPlay={handleDemoLoaded}
          autoPlay
          controls={false}
          start="10"
          type="video/mp4"
          src="./demo.mp4"
        />
      ) : null}
    </section>
  ) : (
    <Loader progress={loadingYolo.progress} />
  );
};

export default ClientRender;
