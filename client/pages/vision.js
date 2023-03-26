import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useRef, useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import Head from "next/head";


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
  const [webcamLoaded, setWebcamLoaded] = useState(false)

  const [peerId, setPeerID] = useState("");
  const peerRef = useRef(null);
  const rtcOutputRef = useRef(null);
  const [WebRTCMode, setWebRTCMode] = useState(false);
  const [WebRTCLoaded, setWebRTCLoaded] = useState(false);

  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);




  useEffect(() => 
  {
 
   
      //Init peerJS
      const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      peerRef.current = newPeer;
      newPeer.on("open", (id) => {
  
        setPeerID(id);
      });

      newPeer.on('connection',  function(conn) {
        console.log('new connection', conn.peer);
        conn.send('Hello!');
      })

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


    

    

  
    }
    initPeerJS();
  }, [])



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
          <div className=" grid relative text-2xl w-full  text-white gap-2 h-20 rounded-xl font-bold bg-orangeFade ">
            <VisionHeader
              WebRTCMode={WebRTCMode}
              peerId={peerId}
              allowWebcam={allowWebcam}
              demo={demo}
              setWebcamL
              webcamLoaded={webcamLoaded}
              setWebcamLoaded={setWebcamLoaded}
              setDemo={setDemo}
              setWebRTCMode={setWebRTCMode}
              setAllowWebcam={setAllowWebcam}  
              handleDisableDemo={handleDisableDemo}
            />

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
            <div className="bg-orangeFade outline-2   outline-black rounded-b-2xl">
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
                webcamLoaded={webcamLoaded}
                webcamPlaying={webcamPlaying}
                setHasWebcam={setHasWebcam}
                setWebRTCMode={setWebRTCMode}
                setAllowWebcam={setAllowWebcam}
                setDemoLoaded={setDemoLoaded}
                setWebcamPlaying={setWebcamPlaying}
                setLoadedCoco={setLoadedCoco}
                setWebcamLoaded={setWebcamLoaded}
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
