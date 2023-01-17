import { useEffect, useRef, useState } from "react";
import { onTakePhotoButtonClick } from "../libs/canvas_utility";
import { capturedImageServer } from "../libs/utillity";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";

const CanvasInput = (props) => {
  const { track, fps, outputRef } = props;


  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
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
        width={imageWidth}
        height={imageHeight}
        className="inline"
      ></canvas>
    </>
  );
};

export default CanvasInput;
