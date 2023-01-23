import { useCallback, useEffect, useState } from "react";
import React from "react";
import Toolbar from "./Toolbar";
import RoisFeed from "./RoisFeed";
import Selector from "./Selector";
import { imageWidthState, imageHeightState } from "./states";
import { useRecoilState } from "recoil";

const Camera = () => {
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState)
  const [track, setTrack] = useState(null);

  const getVideo = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { min: imageWidth } },
    });
   
    const track = stream.getVideoTracks()[0];
    let maxHeight = track.getCapabilities().height.max
    let maxWidth = track.getCapabilities().width.max
    console.log(track.getConstraints());
    setImageWidth(maxWidth)
 
    setTrack(track);
  }, []);
  useEffect(() => {
    if(track === null) { getVideo(); }

  }, [getVideo]);

  return (
    <div className="flex justify-center m-2 border-solid border-4 border-violet-500">
      {/* <Toolbar></Toolbar> */}
      <Selector track={track}></Selector>
      {/* <RoisFeed></RoisFeed> */}
    </div>
  );
};

export default Camera;
