import { useEffect, useRef, useState } from "react";
import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";

const LoadingScreen = ({ setWebRTCMode, setAllowWebcam, allowWebcam, WebRTCMode }) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const enableWebcamRef = useRef(null);
  const [offsetX, setOffsetX] = useState(null)
  const [offsetY, setOffsetY] = useState(null)

  function updateBounding(canvasEl) {
    let canvasOffset = canvasEl.getBoundingClientRect();
    setOffsetX(canvasOffset.left);
    setOffsetY(canvasOffset.top);
  }


  useEffect(() => {
    updateBounding(enableWebcamRef.current);

    if (enableWebcamRef.current !== null) {
      let dotSring = "   ";
      let context = enableWebcamRef.current.getContext("2d");
      context.clearRect(0, 0, imageWidth, imageHeight);
      context.font = "bold 40px Arial";
      context.fillStyle = "blue";
      context.fillRect(0, 0, imageWidth, imageHeight);

      context.textAlign = "center";

      context.clearRect(0, 0, imageWidth, imageHeight);
      context.fillStyle = "black";

      if(allowWebcam) {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        context.fillText(
          "Trying to detect webcam" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.4
        );
      }
      else if(WebRTCMode) 
      {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        context.fillText(
          "Waiting for remote connection" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.4
        );
      }
      else 
      {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        context.fillText(
          "Please Choose a  video source" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.4
        );
      }



    }




  }, [allowWebcam, WebRTCMode]);

  return (
    <>
      <canvas
   
        className="z-10 relative"
        ref={enableWebcamRef}
        height={imageHeight}
        width={imageWidth}
      ></canvas>
    </>
  );
};

export default LoadingScreen;
