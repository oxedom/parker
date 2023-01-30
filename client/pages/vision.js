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
  const [webcamApproved, setWebCamApproved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const [totalFrames, setTotalFrames] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [attempt, setAttempt] = useState(0)




  if (false) {
    return (
      <div className="flex flex-col  p-16 outline outline-1  outline-stone-900">
        <div>
          <div className="flex justify-between border-2 border-black">
            <RoisFeed totalFrames={totalFrames}></RoisFeed>

            {webcamApproved && hasWebcam ? (
              <div className="a">
                <DrawingCanvas></DrawingCanvas>
                <ClientRender
                  loaded={loaded}
                  webcamApprove={webcamApproved}
                  setTotalFrames={setTotalFrames}
                  setLoaded={setLoaded}
                ></ClientRender>
              </div>
            ) : (
              <img
                width={imageWidth}
                height={imageHeight}
                style={{ zIndex: 1 }}
              />
            )}

            <ToolbarTwo
            setAttempt={setAttempt}
              webcamApproved={webcamApproved}
              hasWebcam={hasWebcam}
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
  }
  else 
  {
    return (<EnableWebcam></EnableWebcam>)
  }
};

export default visionPage;
