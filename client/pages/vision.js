import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilValue } from "recoil";
import Head from "next/head";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
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
          <div
            className="flex justify-between rounded-lg outline-3 outline outline-black shadow-lg
            "
          >
            <Toolbar
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
            </Toolbar>
              <div className="">
                <DrawingCanvas setProcessing={setProcessing}></DrawingCanvas>
                <ClientRender
                  hasWebcam={hasWebcam}
                  loadedCoco={loadedCoco}
                  webcamEnabled={webcamEnabled}
                  setHasWebcam={setHasWebcam}
                  setLoadedCoco={setLoadedCoco}
                  setProcessing={setProcessing}
                  processing={processing}
                ></ClientRender>
              </div>
            <RoisFeed></RoisFeed>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default visionPage;
