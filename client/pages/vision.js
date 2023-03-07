import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import { imageWidthState, imageHeightState, selectedRoiState } from "../components/states";
import { useRecoilValue } from "recoil";
import Head from "next/head";
import PreMenu from "../components/PreMenu";
import { totalOccupied } from "../libs/utillity";

const visionPage = () => {
  const [hasWebcam, setHasWebcam] = useState(false);

  const [webcamEnabled, setWebcamEnable] = useState(false);
  const imageWidth = useRecoilValue(imageWidthState);
  const imageHeight = useRecoilValue(imageHeightState);
  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [demo, setDemo] = useState(false);
  const [allowWebcam, setAllowWebcam] = useState(false);
  const selectedRois = useRecoilValue(selectedRoiState);
  const [active, setActivate] = useState(false)
  
  const counts = totalOccupied(selectedRois)
  const openModal = () => {
    setIsModalOpen(true);
  };

  const handleDisableDemo = () => {
    
    setDemo(false);
    // setAllowWebcam(true)
    // setWebcamEnable(true);

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
        <div className="flex flex-col  p-16    ">
          <div className="">

   

              <div className="p-5 relative text-2xl flex items-center   text-white my-4 gap-2 h-20 rounded-lg font-bold bg-orange-600  justify-between" >

              {demo ?
              
              <>
              <h5 className="font-bold p-4 border-white  text-4xl animate-ping duration-300"> DEMO   </h5>
              <span className="" onClick={handleDisableDemo}>
              {" "}
              Click here with your own webcam!
            </span>
            </>
              
              : ""}


              {!demo && !allowWebcam ? 
              <div className="border border-white rounded-lg ">

              <button
                  className="p-3  animate-pulse  align-self-center justify-self-center bg-opacity-70 hover:scale-105 duration-300  rounded-lg ml-2 text-center"
                  onClick={(e) => {
                    setAllowWebcam(true);
                  }}
                >
                  {" "}
                  Enable Webcam{" "}
                </button>

              </div>

              
              
              : ""}

            {!demo && allowWebcam ? 
              <div>
                <div className="grid grid-cols-3 gap-2 justify-center items-center place-content-center self-center"> 
                {/* <h6>{`  spaces: ${selectedRois.length}`}</h6> */}
      <h6>{`  Available : ${counts.availableCount}`}</h6>
      <h6>{`  Occupied : ${counts.OccupiedCount}`}</h6>
                  
                  
                   </div>


              </div>  :""}

                <button  className=" border text-center border-white rounded-md p-4 "> Settings </button>
              </div> 

            <div
              className="hidden md:flex  flex-col md:flex-row  md:justify-between rounded outline-1 outline  outline-black shadow-lg
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
                  allowWebcam={allowWebcam}
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
      )}
    </DashboardLayout>
  );
};

export default visionPage;
