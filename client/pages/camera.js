import Layout from "../layouts/DefaultLayout";

import { useCallback, useEffect, useRef, useState } from "react";

import React from "react";
import CanvasInput from "../components/CanvasInput";
import DrawingCanvas from "../components/DrawingCanvas";

const Camera = () => {
  // renderRectangle.printHello();
  const overlayRef = useRef(null);
  const canvasRef = useRef(null);
  // const inputCanvasRef = React.createRef()
  const videoRef = useRef(null);
  const outputRef = useRef(null);

  const [imageWidth, setImageWidth] = useState(640);
  const [imageHeight, setImageHeight] = useState(480);
  const [renderRectangle, setDrawRectangle] = useState(null);
  const [fps, setFps] = useState(1000);
  const [track, setTrack] = useState(null);

  const getVideo = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { min: imageWidth } },
    });
    const track = stream.getVideoTracks()[0];
    setTrack(track);
  }, []);
  useEffect(() => {
    getVideo();
  }, []);

  return (
    <Layout>


        <div className="flex gap-10">


        <div className="bg-red-500 ">
        <DrawingCanvas imageWidth={imageWidth} imageHeight={imageHeight}>
          {" "}


        </DrawingCanvas>

        <CanvasInput
          track={track}
          fps={fps}
          outputRef={outputRef}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
        />

        </div>



        <img
          className=""
          width={imageWidth}
          height={imageHeight}
          ref={outputRef}
        />

        </div>

    </Layout>
  );
};

export default Camera;
