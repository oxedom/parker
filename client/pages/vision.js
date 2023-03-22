import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useRef, useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import Head from "next/head";
import DisplayInfo from "../components/DisplayInfo";
import Call from "../components/Call";
import { useRouter } from "next/router";
import VisionHeader from "../components/VisionHeader";


const visionPage = () => {



  const [allowWebcam, setAllowWebcam] = useState(false);
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
  const [webcamPlaying, setWebcamPlaying] = useState(false);

  const [demoLoaded, setDemoLoaded] = useState(false);
  const [demo, setDemo] = useState(false);
  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const router = useRouter()
  const {mode }=  router.query

  useEffect(() => 
  {
    if(mode === "demo") 
    {
      setDemo(true)
    }
  }, [])

  const [WebRTCMode, setWebRTCMode] = useState(false)
  const [peerId, setPeerID] = useState("")
  const peerRef = useRef(null)
  const rtcOutputRef = useRef(null)





  const handleDisableDemo = () => {
    setDemo(false);
  };









  return (
    <DashboardLayout>
      <Head>
        <title> Vison</title>
      </Head>

        <div className={`flex flex-col  p-16    `}>
          <div className="flex flex-col justify-center items-center">
            <div className="bg-filler grid relative text-2xl   text-white my-4 gap-2 h-20 rounded-lg font-bold bg-orange-600  ">
              <VisionHeader  WebRTCMode={WebRTCMode} webcamEnabled={webcamEnabled} demo={demo} handleDisableDemo={handleDisableDemo} setWebRTCMode={setWebRTCMode} setAllowWebcam={setAllowWebcam}/>
              {/* {WebRTCMode ?
              
              
                <div>
                  <Call peerId={peerId} remoteVideoRef={rtcOutputRef}></Call>
                             <input className="w-[150px] text-black" value={peerId}  alt="connectID" placeholder="connectID" onChange={(e) => { 
                e.preventDefault()
                setPeerID(e.target.value)}} type="text"/>
                  </div>

   
   
             

    
               : null} */}

              {!demo && allowWebcam ? <div></div> : ""}
   
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
                  WebRTCMode={WebRTCMode}
                  setWebRTCMode={setWebRTCMode}
                  setDemoLoaded={setDemoLoaded}
                  rtcOutputRef={rtcOutputRef}
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

    </DashboardLayout>
  );
};

export default visionPage;
