import { useEffect, useRef, useState } from "react";
import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";

const LoadingScreen = ({
  setWebRTCMode,
  setAllowWebcam,
  allowWebcam,
  WebRTCMode,
}) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const enableWebcamRef = useRef(null);
  const [offsetX, setOffsetX] = useState(null);
  const [offsetY, setOffsetY] = useState(null);

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

      if (allowWebcam) {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        context.fillText(
          "    Attempting to detect webcam" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.4
        );
      } else if (WebRTCMode) {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";

        const txt = `Invite with link \n to make a remote \n video connection`;
        const lines = txt.split("\n");
        const lineheight = 40;

        for (var i = 0; i < lines.length; i++)
          context.fillText(
            lines[i],
            imageWidth / 2,
            imageHeight / 2.5 + i * lineheight
          );
      } else {
        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        const txt =
          "  Choose a video source from the \n navigation bar  \n  \n Demo/Webcam/Remote  ";
        const lines = txt.split("\n");
        const lineheight = 40;
        for (var i = 0; i < lines.length; i++)
          context.fillText(
            lines[i],
            imageWidth / 2,
            imageHeight / 2.5 + i * lineheight
          );
      }
    }
  }, [allowWebcam, WebRTCMode, imageHeight, imageWidth]);

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
