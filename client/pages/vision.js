import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilValue } from "recoil";
import Head from "next/head";
import PreMenu from "../components/PreMenu";


const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);

  const [webcamEnabled, setWebcamEnable] = useState(false);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [demo, setDemo] = useState(false)
  const openModal = () => {
    setIsModalOpen(true);
  };

  const handleDisableDemo = () => 
  {
    setDemo(false)
    setWebcamEnable(true)
  }

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
    
    <DashboardLayout>
      <Head>
        <title> Vison</title>
      </Head>

      {(!demo && !webcamEnabled) ?  <PreMenu setDemo={setDemo} setWebcamEnable={setWebcamEnable} ></PreMenu>: 
     
      <div className="flex flex-col  p-16   ">
        <div className="">

          {demo ? <h4 className="p-5 text-2xl text-white rounded-lg font-bold bg-orange-600 "  > DEMO MODE: <span className="" onClick={handleDisableDemo} > Click here with your own webcam!</span></h4> : <></>}
          <div
            className="hidden md:flex  flex-col md:flex-row  md:justify-between rounded-lg outline-3 outline outline-black shadow-lg
            "
          >
            <Toolbar
              setWebcamEnable={setWebcamEnable}
              setDemo={setDemo}
              demo={demo}
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
                  demo={demo}
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
}
    </DashboardLayout>
  );
};

export default visionPage;
