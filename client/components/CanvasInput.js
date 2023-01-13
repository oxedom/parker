import { useEffect, useRef, useState } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
const CanvasInput = (props) => {
  const { track, fps, outputRef } = props;
  const inputRef = useRef(null);
    const [count, setCount] = useState(0)
  useEffect(() => {
    function intervalProcessing(track) {
      setTimeout(async () => {
        const imageCaptured = new ImageCapture(track);
        onTakePhotoButtonClick(imageCaptured, inputRef);
        const data = await capturedImageServer(imageCaptured);
        // outputRef.current.src = data.img
        setCount((prev) => { return prev+1})
      }, fps);
    }
    if (track !== null) {
      intervalProcessing(track);
    }
  }, [track]);
  return (
    <>
        <canvas
      style={{ zIndex: -9 }}
      ref={inputRef}
      width={props.imageWidth}
      height={props.imageHeight}
      className="absolute"
    ></canvas>
    <h1> {count} </h1>
    </>

  );
};

export default CanvasInput;
