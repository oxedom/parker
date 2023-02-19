import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useState } from "react";
import ToolbarTwo from "../components/Toolbar";

import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilValue } from "recoil";
import EnableWebcam from "../components/EnableWebcam";
import Head from "next/head";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
  const [reload, setReload] = useState(0);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [processing, setProcessing] = useState(true);
  const [showDetections, setShowDetections] = useState(false);

  if (hasWebcam) {
    return (
      <div>
        <Head>
          <title> Vison</title>
        </Head>

        <div className="flex flex-col  p-16  ">
          <div className="">
            <div
              className="flex justify-between rounded-xl border border-gray-500/100  
            shadow-xl"
            >
              <ToolbarTwo
                setReload={setReload}
                setWebcamEnable={setWebcamEnable}
                setProcessing={setProcessing}
                setShowDetections={setShowDetections}
                showDetections={showDetections}
                processing={processing}
                setHasWebcam={setHasWebcam}
                hasWebcam={hasWebcam}
                webcamEnabled={webcamEnabled}
              >
                {" "}
              </ToolbarTwo>

              {webcamEnabled ? (
                <div className="">
                  <DrawingCanvas setProcessing={setProcessing}></DrawingCanvas>
                  <ClientRender
                    showDetections={showDetections}
                    processing={processing}
                  ></ClientRender>
                </div>
              ) : (
                <video
                  width={imageWidth}
                  style={{ width: imageWidth, height: imageHeight }}
                  height={imageHeight}
                />
              )}

              <RoisFeed></RoisFeed>
            </div>
          </div>
        </div>
      </div>
    );
  } else {
    return (
      <EnableWebcam
        setHasWebcam={setHasWebcam}
        hasWebcam={hasWebcam}
        webcamEnabled={webcamEnabled}
        setWebcamEnable={setWebcamEnable}
        reload={reload}
        setReload={setReload}
      ></EnableWebcam>
    );
  }
};

export default visionPage;
