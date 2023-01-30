import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useState } from "react";
import ToolbarTwo from "../components/ToolbarTwo";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";
import EnableWebcam from "../components/EnableWebcam";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);

  const [webcamApproved, setWebCamApproved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [reload, setReload] = useState(0)
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [totalFrames, setTotalFrames] = useState(0);

  const [processing, setProcessing] = useState(false);


  if (hasWebcam) {
    return (
      <div className="flex flex-col  p-16 outline outline-1  outline-stone-900">
        <div>
          <div className="flex justify-between border-2 border-black">
            <RoisFeed totalFrames={totalFrames}></RoisFeed>

   
              <div className="">
                <DrawingCanvas></DrawingCanvas>
                <ClientRender
                  loaded={loaded}
                  webcamApprove={webcamApproved}
                  setTotalFrames={setTotalFrames}
                  setLoaded={setLoaded}
                ></ClientRender>
              </div>
    
  

            <ToolbarTwo
              setReload={setReload}
              setWebCamApproved={setWebCamApproved}
              setProcessing={setProcessing}
              processing={processing}
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
    ></EnableWebcam>;
  }
};

export default visionPage;
