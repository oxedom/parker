import { useCallback, useEffect, useState, useRef } from "react";
import React from "react";
import Selector from "./Selector";
import {
  imageWidthState,
  imageHeightState,
  trackState,
  processingState,
} from "./states";
import { useRecoilState, useRecoilValue } from "recoil";

const Camera = () => {
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [track, setTrack] = useRecoilState(trackState);
  const processing = useRecoilValue(processingState);
  // const [imageHeight, setImageHeight] = useRecoilState(imageHeightState)

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
  }, [processing]);

  return (
    <div className="flex justify-center m-2 ">
      <Selector></Selector>
    </div>
  );
};

export default Camera;
