import { useEffect, useRef } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
const Canvas = (props) => {
  const { track, fps, outputRef } = props;
  const inputRef = useRef(null);

  useEffect(() => {
    function intervalProcessing(track) {
      setInterval(async () => {
        const imageCaptured = new ImageCapture(track);
        onTakePhotoButtonClick(imageCaptured, inputRef);
        const data = await capturedImageServer(imageCaptured);
        // outputRef.current.src = data.img
      }, fps);
    }
    if (track !== null) {
      intervalProcessing(track);
    }
  }, [track]);
  return (
    <canvas
      style={{ zIndex: -9 }}
      ref={inputRef}
      width={props.imageWidth}
      height={props.imageHeight}
      className="absolute"
    ></canvas>
  );
};

export default Canvas;
