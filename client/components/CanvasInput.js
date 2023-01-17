import { useEffect, useRef, useState } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
import {
  imageWidthState,
  imageHeightState,
  processingState,
} from "../components/states";
import {  useRecoilValue } from "recoil";

const CanvasInput = (props) => {
  const { track, fps } = props;

  const processing = useRecoilValue(processingState);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);

  const [imgState, setImageState] = useState("");
  const inputRef = useRef(null);

  const detectionsRef = useRef(null);
  let dectXRef = useRef(null);

  let clientFPS = 10
  let serverFPS = 1000

  let processingId;
  let renderingId;


  function renderAllDetections(detections)
  {
    dectXRef.current.clearRect(
      0,
      0,
      inputRef.current.width,
      inputRef.current.height
    );

      detections.forEach(d => { renderDetction(d)})
  }



  function renderDetction(d) 
  {
    const {height,right_x,top_y,width} = d.cords
   dectXRef.current.strokeRect(right_x, top_y, width,height)
  
   

 


  }


  useEffect(() => 
  {
    const detectionsEl = detectionsRef.current
    dectXRef.current = detectionsEl.getContext("2d");
    const dtx = dectXRef.current;


    dtx.strokeStyle = "#78E3FD";
    dtx.lineWidth = 10;




  },[])




  useEffect(() => {
    function renderWebcam(track) {
      renderingId = setInterval(() => {
   
        const imageCaptured = new ImageCapture(track);
        onTakePhotoButtonClick(imageCaptured, inputRef);
      }, clientFPS);
    }

    function renderProcess(track) {
      processingId = setInterval(async () => {
        const imageCaptured = new ImageCapture(track);
        onTakePhotoButtonClick(imageCaptured, inputRef);
        const data = await capturedImageServer(imageCaptured);
        const {detections} = (data.meta_data)
        renderAllDetections(detections)

        setImageState(data.img);
      }, serverFPS);
    }

    if (track !== null && !processing) {
      renderWebcam(track);
    }
    if (track !== null && processing) {
      renderProcess(track);
    }

    return () => {
      clearInterval(renderingId);
      clearInterval(processingId);
    };
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
        // style={{ zIndex: 1}}
        id="input-imagebitmap"
        ref={inputRef}
        width={imageWidth}
        height={imageHeight}
        className="inline"
      ></canvas> 

      {/* {processing && <img    width={imageWidth}
        height={imageHeight} className="inline" src={imgState} /> } */}
    </>
  );
};

export default CanvasInput;
