import { useEffect, useRef } from "react";
import { imageCapturedToCanvas, renderRoi } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
import {
  imageWidthState,
  imageHeightState,
  processingState,
  detectionColorState
} from "../components/states";
import { useRecoilValue } from "recoil";


const CanvasInput = ({ track }) => {
  //Fetching from recoil store using atoms
  const processing = useRecoilValue(processingState);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);

  const detectionColor = useRecoilValue(detectionColorState)

  //Ref declaring
  const inputRef = useRef(null);
  const detectionsRef = useRef(null);
  let dectXRef = useRef(null);

  //FPS declaring need to be a STATE
  let clientFPS = 10;
  let serverFPS = 1000;

  //init vars for interval ID's
  let processingId;
  let renderingId;

  function renderAllDetections(detections) {
    //Clears canvas before rendering all overlays (Runs each response)
    clearDetectionOverlay();
    //For each on the detections
    detections.forEach((d) => {
      console.log(detectionColor);
      renderRoi(d, dectXRef, detectionColor);
    });
  }

  function clearDetectionOverlay() {
    //Clears canvas
    dectXRef.current.clearRect(
      0,
      0,
      inputRef.current.width,
      inputRef.current.height
    );
  }


  useEffect(() => {
    //Need to do this for canvas2d to work
    const detectionsEl = detectionsRef.current;
    dectXRef.current = detectionsEl.getContext("2d");
    const dtx = dectXRef.current;

    dtx.strokeStyle = "#78E3FD";
  }, []);

  useEffect(() => {
    function renderWebcam(track) {
      renderingId = setInterval(() => {
        const imageCaptured = new ImageCapture(track);
        //Renders the imageCaptured into a canvas
        imageCapturedToCanvas(imageCaptured, inputRef);
        //Speed can be very high
      }, clientFPS);
    }

    function renderProcess(track) {
      processingId = setInterval(async () => {
        const imageCaptured = new ImageCapture(track);
        imageCapturedToCanvas(imageCaptured, inputRef);
        const data = await capturedImageServer(imageCaptured);
        let { detections } = data.meta_data;
        console.log(detections);
        detections = detections.map(d => ({...d, color: detectionColor }))
        renderAllDetections(detections);

        //Speed SHOULD BE min server capacity
      }, serverFPS);
    }

    //Clears canvas
    clearDetectionOverlay();

    //If there is a track and the processing mode is toogled to false
    // so only render the users webcam locally
    if (track !== null && !processing) {
      renderWebcam(track);
    }

    ////If there is a track and the processing mode is  toogled true
    //render to client and send the webcam output to the server
    if (track !== null && processing) {
      renderProcess(track);
    }

    return () => {
      //Clears intervals when component unmounts
      clearInterval(renderingId);
      clearInterval(processingId);
    };

    //Runs when processing is toogled or track changes
  }, [track, processing, detectionColor]);

  return (
    <>
      <canvas
        id="detections-overlay"
        ref={detectionsRef}
        width={imageWidth}
        height={imageHeight}
        className="fixed"
      ></canvas>

      <canvas
        id="input-imagebitmap"
        ref={inputRef}
        width={imageWidth}
        height={imageHeight}
        className="inline"
      ></canvas>
    </>
  );
};

export default CanvasInput;
