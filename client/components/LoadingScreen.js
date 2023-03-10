import { useEffect, useRef } from "react";
import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";
const LoadingScreen = () => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);

  const enableWebcamRef = useRef(null);
  useEffect(() => {
    let loadingIntervalID;
    if (enableWebcamRef.current !== null) {
      let dotSring = "   ";
      let context = enableWebcamRef.current.getContext("2d");
      context.clearRect(0, 0, imageWidth, imageHeight);
      context.font = "bold 40px Arial";
      context.fillStyle = "blue";
      context.fillRect(0, 0, imageWidth, imageHeight);

      context.textAlign = "center";
      loadingIntervalID = setInterval(() => {
        if (dotSring === "   ") {
          dotSring = ".  ";
        } else if (dotSring === ".  ") {
          dotSring = ".. ";
        } else if (dotSring === ".. ") {
          dotSring = "...";
        } else if (dotSring === "...") {
          dotSring = "   ";
        }

        context.clearRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "blue";

        context.fillRect(0, 0, imageWidth, imageHeight);
        context.fillStyle = "white";
        context.fillText(
          "Please enable your webcam" + dotSring,
          imageWidth * 0.5,
          imageHeight * 0.3
        );
        context.font = "bold 28px Arial";
        context.fillText(
          "Make sure it's plugged in!   ",
          imageWidth * 0.5,
          imageHeight * 0.3 + 50
        );
        context.fillText(
          "For Troubleshooting check the docs",
          imageWidth * 0.5,
          imageHeight * 0.3 + 100
        );
        context.font = "bold 40px Arial";
      }, 1500);
    }

    return () => {
      clearInterval(loadingIntervalID);
    };
  }, []);

  return (
    <canvas
      ref={enableWebcamRef}
      height={imageHeight}
      width={imageWidth}
    ></canvas>
  );
};

export default LoadingScreen;
