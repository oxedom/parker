import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useRef, useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import Head from "next/head";


import { useRouter } from "next/router";
import VisionHeader from "../components/VisionHeader";
import VisionFooter from "../components/VisionFooter";
import { createEmptyStream } from "../libs/webRTC_utility";
import { useRecoilState } from "recoil";
import {
  imageWidthState,
  imageHeightState,

} from "../components/states"
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
  const router = useRouter();
  const { mode } = router.query;


  const [peerId, setPeerID] = useState("");
  const peerRef = useRef(null);
  const rtcOutputRef = useRef(null);
  const [WebRTCMode, setWebRTCMode] = useState(false);
  const [WebRTCLoaded, setWebRTCLoaded] = useState(false);

  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);

  useEffect(() => {
    if (mode === "demo") {
      setDemo(true);
    }
  }, []);


  useEffect(() => 
  {
 
    if(WebRTCMode) 
    {
      //Init peerJS
      const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      peerRef.current = newPeer;
      newPeer.on("open", (id) => {
  
        setPeerID(id);
      });

      newPeer.on("call", (call) => {
        let fakeStream = createEmptyStream()
        call.answer(fakeStream);
        console.log("im getting a call ");
        call.on('stream', (remoteStream) => 
        {

          
          rtcOutputRef.current.srcObject = remoteStream;
          rtcOutputRef.current.play();
        })
      });


    

    };

    initPeerJS();
    }

  }, [WebRTCMode])



  const handleDisableDemo = () => {
    setDemo(false);
    setImageWidth(640)
    setImageHeight(480)
   
  };

  return (
    <DashboardLayout>
      <Head>
        <title> Vison</title>
      </Head>

      <div className={`flex flex-col  p-16   `}>
        <div className="flex flex-col justify-center items-center gap-4">
          <div className=" grid relative text-2xl w-full  text-white gap-2 h-20 rounded-xl font-bold bg-orange-500  ">
            <VisionHeader
              WebRTCMode={WebRTCMode}
              webcamEnabled={webcamEnabled}
              demo={demo}
              setWebRTCMode={setWebRTCMode}
              setAllowWebcam={setAllowWebcam}  
              handleDisableDemo={handleDisableDemo}
            />
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
            className={`hidden md:flex gap-4 p-2  flex-col md:flex-row  md:justify-between 
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
            <div className="bg-orange-500 outline-2   outline-black rounded-b-2xl">
              {demo || webcamPlaying || WebRTCLoaded ? <DrawingCanvas ></DrawingCanvas> : null}
 
              <ClientRender
                demo={demo}
                hasWebcam={hasWebcam}
                loadedCoco={loadedCoco}
                allowWebcam={allowWebcam}
                webcamEnabled={webcamEnabled}
                processing={processing}
                WebRTCMode={WebRTCMode}
                WebRTCLoaded={WebRTCLoaded}
                rtcOutputRef={rtcOutputRef}
                demoLoaded={demoLoaded}
                webcamPlaying={webcamPlaying}
                setHasWebcam={setHasWebcam}
                setWebRTCMode={setWebRTCMode}
                setAllowWebcam={setAllowWebcam}
                setDemoLoaded={setDemoLoaded}
                setWebcamPlaying={setWebcamPlaying}
                setLoadedCoco={setLoadedCoco}
                setProcessing={setProcessing}
                setWebRTCLoaded={setWebRTCLoaded}
              ></ClientRender>
               <VisionFooter  WebRTCMode={WebRTCMode} WebRTCLoaded={WebRTCLoaded} peerId={peerId} > </VisionFooter> 
              
            </div>
            <RoisFeed></RoisFeed>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default visionPage;
