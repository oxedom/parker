import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useState } from "react";
import ToolbarTwo from "../components/ToolbarTwo";
import { imageWidthState, imageHeightState } from "../components/states";
import {  useRecoilValue } from "recoil";
import EnableWebcam from "../components/EnableWebcam";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
  const [reload, setReload] = useState(0)
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [processing, setProcessing] = useState(true)




  if (hasWebcam) {
    return (
      <div className="flex flex-col  p-16 ">
        <div>
          <div className="flex justify-between border-2 border-black">
            <RoisFeed ></RoisFeed>

            {webcamEnabled 
              ? (
                <div className="">
                  <DrawingCanvas></DrawingCanvas>
                  <ClientRender
                  processing={processing}
                  ></ClientRender>
                </div>
              )
              : <video width={imageWidth} style={{width:imageWidth, height:imageHeight}} height={imageHeight}/>
            }


    
  

            <ToolbarTwo
              setReload={setReload}
              setWebcamEnable={setWebcamEnable}
              setProcessing={setProcessing}
              processing={processing}
              setHasWebcam={setHasWebcam}
              hasWebcam={hasWebcam}
              webcamEnabled={webcamEnabled}
            >
              {" "}
            </ToolbarTwo>
          </div>
        </div>
      </div>
    );
  } else {
    return <EnableWebcam
    setHasWebcam={setHasWebcam}
    hasWebcam={hasWebcam}
    webcamEnabled={webcamEnabled}
    setWebcamEnable={setWebcamEnable}
    reload={reload}
    setReload={setReload}
    ></EnableWebcam>;
  }
};

export default visionPage;
