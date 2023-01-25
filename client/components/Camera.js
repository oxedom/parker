import { useCallback, useEffect, useState, useRef } from "react";
import React from "react";
import Selector from "./Selector";
import { imageWidthState, imageHeightState } from "./states";
import { useRecoilState } from "recoil";
import { predictWebcam, loadModel } from "../libs/utillity";
const Camera = () => {
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  // const [imageHeight, setImageHeight] = useRecoilState(imageHeightState)
  const [track, setTrack] = useState(null);

  const getVideo = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { min: imageWidth } },
    });

    const track = stream.getVideoTracks()[0];
    let maxWidth = track.getCapabilities().width.max;

    setImageWidth(maxWidth);
    setTrack(track);
  }, []);

  useEffect(() => {
    if (track === null) {
      getVideo();
    }
  }, []);

  return (
    <div className="flex justify-center m-2 ">
      <Selector track={track}></Selector>
    </div>
  );
};

export default Camera;
