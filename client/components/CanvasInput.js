import { useEffect, useRef, useState } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";


const CanvasInput = (props) => {
  const { track, fps, outputRef } = props;
  const inputRef = useRef(null);


  useEffect(() => {
    function intervalProcessing(track) {
      setTimeout(async () => {
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
    <>
      <canvas
        // style={{ zIndex: 1}}
        id="input-imagebitmap"
        ref={inputRef}
        width={props.imageWidth}
        height={props.imageHeight}
        className="inline"
      ></canvas>

    </>
  );
};

export default CanvasInput;
