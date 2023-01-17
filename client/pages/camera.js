import Layout from "../layouts/DefaultLayout";

import { useCallback, useEffect, useRef, useState } from "react";
import React from "react";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import Selector from "../components/Selector";
import { imageWidthState, processingState } from "../components/states";
import { useRecoilState } from "recoil";

const Camera = () => {
  const [imageWidth] = useRecoilState(imageWidthState);
  const [processing, setProcessing] = useRecoilState(processingState);

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
  }, [getVideo]);

  return (
    <Layout>
      <div className="flex justify-center m-2">
        <Toolbar fps={fps}></Toolbar>
        {true ? (
          <>
            <Selector fps={fps} track={track}></Selector>
          </>
        ) : (
          <></>
        )}

        <RoisFeed></RoisFeed>
      </div>
    </Layout>
  );
};

export default Camera;
