import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import Head from "next/head";
import PreMenu from "../components/PreMenu";
import DisplayInfo from "../components/DisplayInfo";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);

  const [demoLoaded, setDemoLoaded] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);

  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [demo, setDemo] = useState(false);
  const [allowWebcam, setAllowWebcam] = useState(false);
  const [webcamPlaying, setWebcamPlaying] = useState(false);
  const [active, setActivate] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const handleDisableDemo = () => {
    setDemo(false);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
    <DashboardLayout>
      <Head>
        <title> Vison</title>
      </Head>

      {!demo && !active ? (
        <PreMenu setDemo={setDemo} setActivate={setActivate}></PreMenu>
      ) : (
        <div
          className={`flex flex-col ${
            demo && !demoLoaded ? "opacity-0 " : "opacity-100"
          } p-16    `}
        >
          <div className="flex flex-col justify-center items-center">
            <div className="bg-filler relative text-2xl grid grid-cols-3   place-items-center items-center center   text-white my-4 gap-2 h-20 rounded-lg font-bold bg-orange-600  ">
              {demo ? (
                <div onClick={handleDisableDemo}>
                  <h5 className="font-bold p-4 rounded-lg text-gray-200 justify-self-start hover:text-white bg-orange-600   text-2xl animate-ping duration-300  ">
                    {" "}
                    Exit{" "}
                  </h5>
                  <span className=""> </span>
                </div>
              ) : (
                ""
              )}

              {!demo && !allowWebcam ? (
                <div className="border animate-pulse transition ease-in animate-indicate border-white rounded-lg ">
                  <button
                    className="p-3  duration-150 align-self-center justify-self-center  rounded-lg text-center"
                    onClick={(e) => {
                      setAllowWebcam(true);
                    }}
                  >
                    {" "}
                    <span className="hover:animate-spin ">Enable Webcam{" "}</span>
                    
                  </button>
                </div>
              ) : (
                ""
              )}

              {!demo && allowWebcam ? <div></div> : ""}
              <DisplayInfo></DisplayInfo>
              <button onClick={openModal} className=" border text-center border-white rounded-md p-4 ">
                {" "}
                Settings{" "}
              </button>
            </div>

            <div
              className={`hidden md:flex    flex-col md:flex-row  md:justify-between rounded outline-1 outline  outline-black shadow-lg
            `}
            >
              <Toolbar
     
           
                setProcessing={setProcessing}
        
                isModalOpen={isModalOpen}
                allowWebcam={allowWebcam}
                setAllowWebcam={setAllowWebcam}
                closeModal={closeModal}
   
                processing={processing}
                hasWebcam={hasWebcam}
        
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
                  allowWebcam={allowWebcam}
                  webcamEnabled={webcamEnabled}
                  setHasWebcam={setHasWebcam}
                  setDemoLoaded={setDemoLoaded}
                  demoLoaded={demoLoaded}
                  webcamPlaying={webcamPlaying}
                  setWebcamPlaying={setWebcamPlaying}
                  setLoadedCoco={setLoadedCoco}
                  setProcessing={setProcessing}
                  processing={processing}
                ></ClientRender>
              </div>
              <RoisFeed></RoisFeed>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
};

export default visionPage;
