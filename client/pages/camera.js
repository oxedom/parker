import Layout from "../layouts/DefaultLayout";
import Canvas from "../components/Canvas";
import Video from "../components/Video";
import { useEffect, useRef, useState } from "react";
import {
  onTakePhotoButtonClick,
  renderRectangleFactory,
} from "../libs/canvas_utility";
import React from "react";
import CanvasInput from "../components/CanvasInput";

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
  const [fps, setFps] = useState(2000);
  const [track, setTrack] = useState(null);

  useEffect(() => {
    const getVideo = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { min: imageWidth } },
      });

      //Gets the current screen

      const track = stream.getVideoTracks()[0];

      setTrack(track);

      // let { width, height } = track.getSettings();
    };
    getVideo();
    console.log(canvasRef);
    // setDrawRectangle(renderRectangleFactory(canvasRef.current, overlayRef.current))
  }, []);

  return (
    <Layout>
      <div className="relative">
        <h2> I want to draw on you</h2>
        <Canvas
          ref={overlayRef}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
        />
        <Canvas
          ref={canvasRef}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
        />
        <CanvasInput
          track={track}
          fps={fps}
          outputRef={outputRef}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
        />
      </div>
      <video
        className="invisible"
        ref={videoRef}
        width={imageWidth}
        height={imageHeight}
      ></video>
      {/* <img  className="none" width={imageWidth} height={imageHeight} ref={outputRef}/> */}
    </Layout>
  );
};

export default Camera;
