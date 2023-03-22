import { useEffect, useRef, useState } from "react";
import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";

const LoadingScreen = ({ setWebRTCLoaded, setAllowWebcam }) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [offsetX, setOffsetX] = useState(undefined);
  const [offsetY, setOffsetY] = useState(undefined);
  const [direction, setDirection] = useState("");
  const enableWebcamRef = useRef(null);

  function updateBounding(canvasEl) {
    let canvasOffset = canvasEl.getBoundingClientRect();
    setOffsetX(canvasOffset.left);
    setOffsetY(canvasOffset.top);
  }

  const handleClick = () => {
    if (direction == "left") {
      setWebRTCLoaded(false);
      setAllowWebcam(true);
    } else {
      setWebRTCLoaded(true);
      setAllowWebcam(false);
    }
  };
  const handleMouseOver = (e) => {
    let x = parseInt(e.clientX - offsetX);
    let y = parseInt(e.clientY - offsetY);
    if (x > 0 && x < imageWidth / 2) {
      setDirection("left");
    } else if (x > 0 && x > imageWidth / 2 && x < imageWidth + 1) {
      setDirection("right");
    }
  };
  useEffect(() => {
    updateBounding(enableWebcamRef.current);
    let loadingIntervalID;
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

      context.fillRect(0, 0, imageWidth, imageHeight);
      context.fillStyle = "white";
      context.fillText(
        "Choose a input mode" + dotSring,
        imageWidth * 0.5,
        imageHeight * 0.3
      );
      context.font = "bold 28px Arial";
      context.fillStyle = "green";

      context.fillRect(imageWidth / 6, 200, 150, 70);
      context.fillStyle = "red";
      context.fillRect(imageWidth / 2, 200, 150, 70);
      context.fillStyle = "white";

      context.fillText("Webcam", imageWidth * 0.29, imageHeight * 0.3 + 100);

      context.fillText(`Stream`, imageWidth * 0.6, imageHeight * 0.3 + 100);

      context.font = "bold 40px Arial";
    }

    return () => {
      clearInterval(loadingIntervalID);
    };
  }, []);

  return (
    <>
      <canvas
        onClick={handleClick}
        onMouseMove={handleMouseOver}
        className="z-10 relative"
        ref={enableWebcamRef}
        height={imageHeight}
        width={imageWidth}
      ></canvas>
    </>
  );
};

export default LoadingScreen;
