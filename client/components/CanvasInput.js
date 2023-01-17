import { useEffect, useRef } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
import { imageWidthState, imageHeightState, outputImageState, processingState} from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";

const CanvasInput = (props) => {
  const { track, fps } = props;


  const processing = useRecoilState(  processingState)
  const [outputImage, setOutputImage] = useRecoilState(outputImageState)
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const inputRef = useRef(null);
  const outputRef = useRef(null)
  let processingId;
  let renderingId


  function renderWebcamToCanvas(track) 
  {
    const imageCaptured = new ImageCapture(track);
    onTakePhotoButtonClick(imageCaptured, inputRef);
  }




useEffect(() => {

  function intervalProcessing(track) {
    processingId = setInterval(async () => {
    const imageCaptured = new ImageCapture(track);
     const data = await capturedImageServer(imageCaptured);
    
   }), fps};  


   function intervalRender(track) 
   {
    renderingId = setInterval(async () => {
      renderWebcamToCanvas(track)

    
    }, fps);
   }


  
  if(processing && track !== null)
  {
    //If processing so you should draw the DATA on the canvas and reRender webcam locally all the time
    intervalProcessing(track)


  }
  if(!processing && track !== null){
    //If not processing you should only render webcam locally

    intervalRender(track)

  }

  


  return () => {
    clearInterval(processingId);
    clearInterval(renderingId);
  };
}, [track]);



  return (
    <>
      <canvas
        // style={{ zIndex: 1}}
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
