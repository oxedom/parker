import Layout from "../layouts/DefaultLayout";

import { useCallback, useEffect, useRef, useState } from "react";
import React from "react";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { finalName } from "../libs/utillity";
import Selector from "../components/Selector";
import { imageWidthState, selectedRoiState} from "../components/states";
import { useRecoilState } from "recoil";

const Camera = () => {
  const outputRef = useRef(null);


  const [imageWidth] = useRecoilState(imageWidthState);

  const [fps, setFps] = useState(1000);
  const [track, setTrack] = useState(null);
  const [processing, setProcessing] = useState(false);


  const [selectingBoxColor, setSelectingBoxColor] = useState("#8FC93A");
  const [selectedBoxColor, setSelectedBoxColor] = useState("#EC3945");



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
      {!processing ? (
        <div className="flex justify-center m-2  ">
          <>
            <Toolbar
  
              fps={fps}
        
            ></Toolbar>
            <div className="border-t-8  border-gray-200">
              <Selector
     
                outputRef={outputRef}
                fps={fps}
                track={track}
      
              >


              </Selector>
            </div>
           
            <RoisFeed ></RoisFeed>
          </>
        </div>
      ) : (
        <></>
      )}
    </Layout>
  );
};

export default Camera;
