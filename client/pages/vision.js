import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useState } from "react";
import ToolbarTwo from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import {
  imageWidthState,
  imageHeightState,
  fpsState,
} from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";
import EnableWebcam from "../components/EnableWebcam";
import Head from "next/head";
import AppNavbar from "../components/AppNavbar";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
  const [reload, setReload] = useState(0);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
    <DashboardLayout>
      <Head>
        <title> Vison</title>
      </Head>

      <div className="flex flex-col  p-16   ">
        <div className="">
          <AppNavbar openModal={openModal}></AppNavbar>
          <div
            className="flex justify-between rounded-lg outline-3 outline outline-black shadow-lg
            "
          >
            <ToolbarTwo
              setReload={setReload}
              setWebcamEnable={setWebcamEnable}
              setProcessing={setProcessing}
      
              setHasWebcam={setHasWebcam}
              isModalOpen={isModalOpen}
              closeModal={closeModal}
              openModal={openModal}

              processing={processing}
              hasWebcam={hasWebcam}
              webcamEnabled={webcamEnabled}
              loadedCoco={loadedCoco}
            >
              {" "}
            </ToolbarTwo>
            {!hasWebcam ? (
              <EnableWebcam
                setHasWebcam={setHasWebcam}
                hasWebcam={hasWebcam}
                webcamEnabled={webcamEnabled}
                setWebcamEnable={setWebcamEnable}
                reload={reload}
                setReload={setReload}
              ></EnableWebcam>
            ) : (
              <></>
            )}
            {webcamEnabled && hasWebcam ? (
              <div className="">
                <DrawingCanvas setProcessing={setProcessing}></DrawingCanvas>
                <ClientRender
                  loadedCoco={loadedCoco}
                  setLoadedCoco={setLoadedCoco}
                  setProcessing={setProcessing}
  
                  processing={processing}
                ></ClientRender>
              </div>
            ) : (
              <video
                width={imageWidth}
                className={`bg-yellow-400 ${!hasWebcam ? "hidden" : "block"}`}
                style={{ width: imageWidth, height: imageHeight }}
                height={imageHeight}
              ></video>
            )}

            <RoisFeed></RoisFeed>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
  // } else {
  //   return (
  //     <DashboardLayout>
  //     {/* <EnableWebcam
  //       setHasWebcam={setHasWebcam}
  //       hasWebcam={hasWebcam}
  //       webcamEnabled={webcamEnabled}
  //       setWebcamEnable={setWebcamEnable}
  //       reload={reload}
  //       setReload={setReload}
  //     ></EnableWebcam> */}
  //           </DashboardLayout>
  //   );
  // }
};

export default visionPage;
