import { useEffect, useRef, useState } from "react";
import { imageCapturedToCanvas, renderRoi } from "../../libs/canvas_utility";
import { capturedImageServer, checkOverlapArrays } from "../../libs/utillity";
import {
  imageWidthState,
  imageHeightState,
  processingState,
  selectedRoiState,
  detectionColorState,
  trackState,
} from "../states";
import { useRecoilValue, useRecoilState } from "recoil";

const CanvasInput = () => {
  //Fetching from recoil store using atoms
  const processing = useRecoilValue(processingState);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const track = useRecoilValue(trackState);
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);

  const detectionColor = useRecoilValue(detectionColorState);

  //Ref declaring
  const inputRef = useRef(null);
  const detectionsRef = useRef(null);
  const overlapRef = useRef(null);

  let dectXRef = useRef(null);
  let overlayXRef = useRef(null);

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
      renderRoi(d, dectXRef, detectionColor);
    });
  }

  function renderAllOverlaps(overlaps) {
    //Clears canvas before rendering all overlays (Runs each response)
    //For each on the detections
    overlaps.forEach((o) => {
      renderRoi(o, dectXRef, "#FFFF00");
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
    const overlayEl = overlapRef.current;
    overlayXRef.current = overlayEl.getContext("2d");
    dectXRef.current = detectionsEl.getContext("2d");
    const otx = overlayXRef.current;
    const dtx = dectXRef.current;
  }, []);

  useEffect(() => {
    function renderWebcam() {
      renderingId = setInterval(() => {
        if (track != null) {
          const imageCaptured = new ImageCapture(track);

          //Renders the imageCaptured into a canvas
          imageCapturedToCanvas(imageCaptured, inputRef);
          //Speed can be very high
        }
      }, clientFPS);
    }

    function renderProcess() {
      processingId = setInterval(async () => {
        const imageCaptured = new ImageCapture(track);
        imageCapturedToCanvas(imageCaptured, inputRef);
        const data = await capturedImageServer(imageCaptured);
        let { detections } = data.meta_data;

        detections = detections.map((d) => ({ ...d, color: detectionColor }));

        renderAllDetections(detections);

        let overlaps = checkOverlapArrays(detections, selectedRegions);
        renderAllOverlaps(overlaps);

        //Speed SHOULD BE min server capacity
      }, serverFPS);
    }

    //Clears canvas
    clearDetectionOverlay();

    //If there is a track and the processing mode is toogled to false
    // so only render the users webcam locally
    if (track !== null && !processing) {
      renderWebcam();
    }

    ////If there is a track and the processing mode is  toogled true
    //render to client and send the webcam output to the server
    if (track !== null && processing) {
      renderProcess();
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
        id="overlap-overlay"
        ref={overlapRef}
        width={imageWidth}
        height={imageHeight}
        className="fixed"
      ></canvas>

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
