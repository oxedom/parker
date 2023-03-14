import { imageHeightState, imageWidthState } from "./states";
import { useRecoilValue } from "recoil";
import { useEffect, useRef } from "react";

const Loader = ({ progress }) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const loaderRef = useRef(null);

  useEffect(() => {
    let loadingIntervalID;
    if (loaderRef.current !== null) {
      let context = loaderRef.current.getContext("2d");
      context.clearRect(0, 0, imageWidth, imageHeight);
      context.font = "bold 40px Arial";
      context.fillStyle = "blue";
      context.fillRect(0, 0, imageWidth, imageHeight);

      context.textAlign = "center";

      context.clearRect(0, 0, imageWidth, imageHeight);
      context.fillStyle = "black";

      context.fillRect(0, 0, imageWidth, imageHeight);
      context.fillStyle = "white";
      context.fillText(`Loading Model...`, imageWidth * 0.5, imageHeight * 0.3);

      const progressString = `${Math.round(progress) * 100}% `;
      context.fillText(progressString, imageWidth * 0.5, imageHeight * 0.5);
    }

    return () => {
      clearInterval(loadingIntervalID);
    };
  }, [progress]);

  return (
    <canvas ref={loaderRef} width={imageWidth} height={imageHeight}></canvas>
  );
};

export default Loader;
