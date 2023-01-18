import { useEffect, useRef, useState } from "react";
import { imageCapturedToCanvas } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
import {
  imageWidthState,
  imageHeightState,
  processingState,
} from "../components/states";
import { useRecoilValue } from "recoil";

const CanvasInput = ({ track }) => {
  //Fetching from recoil store using atoms
  const processing = useRecoilValue(processingState);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);

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
      renderDetction(d);
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

  function renderDetction(d) {
    //Cords
    const { height, right_x, top_y, width } = d.cords;
    const dtx = dectXRef.current;
    //Gets centerX
    const centerX = right_x + width / 2;
    //Font and Size needs to be state
    dectXRef.current.font = "72px Courier";
    dectXRef.current.textAlign = "center";
    //Draws a rect on the detection
    dtx.strokeStyle = "#B22222";
    dtx.lineWidth = 10;
    dectXRef.current.strokeRect(right_x, top_y, width, height);

    dtx.lineWidth = 11;
    dtx.strokeStyle = "#000000";
    dectXRef.current.strokeRect(right_x - 8, top_y, width, height);

    dectXRef.current.fillText(d.label, centerX, top_y * 0.8);
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
        const { detections } = data.meta_data;

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
  }, [track, processing]);

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
