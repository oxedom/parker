import Layout from "../layouts/DefaultLayout";
import uniqid from "uniqid";
import { useCallback, useEffect, useRef, useState } from "react";
import React from "react";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { finalName } from "../libs/utillity";
import Selector from "../components/Selector";
import {imageWidthState, imageHeightState} from '../components/states'
import { useRecoilState, useRecoilValue } from "recoil";

const Camera = () => {
  const outputRef = useRef(null);

  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState)
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState)


  const [fps, setFps] = useState(1000);
  const [track, setTrack] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [selected, setSelected] = useState([]);
  const [roiType, setRoiType] = useState("Any");
  const [roiName, setRoiName] = useState("");
  const [selectingBoxColor, setSelectingBoxColor] = useState("#8FC93A");
  const [selectedBoxColor, setSelectedBoxColor] = useState("#EC3945");






 
  function handleNewRoi(recentRoi) {
    const selectedObj = {
      name: finalName(roiName, selected.length),
      roi_type: roiType,
      ...recentRoi,
      uid: uniqid(),
    };

    setSelected([...selected, selectedObj]);
  }

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
              setRoiName={setRoiName}
              fps={fps}
              setFps={setFps}
              setRoiType={setRoiType}
              roiName={roiName}
              selectingBoxColor={selectingBoxColor}
              selectedBoxColor={selectedBoxColor}
              setSelectingBoxColor={setSelectingBoxColor}
              setSelectedBoxColor={setSelectedBoxColor}
            ></Toolbar>
            <div className="border-t-8  border-gray-200">
              <Selector 
              selectedBoxColor={selectedBoxColor} 
              selectingBoxColor={selectingBoxColor}
           
              selected={selected}
              outputRef={outputRef}
              fps={fps}
              track={track}
              handleNewRoi={handleNewRoi}
 
       
              ></Selector>
            </div>
            <RoisFeed selected={selected}></RoisFeed>
          </>
        </div>
      ) : (
        <>
 
        </>
      )}
    </Layout>
  );
};

export default Camera;
