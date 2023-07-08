import ClientRender from "../components/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import RoisFeed from "../components/RoisFeed";
import { useEffect, useRef, useState } from "react";
import Toolbar from "../components/Toolbar";
import DashboardLayout from "../layouts/DashboardLayout";
import Head from "next/head";
import DataManger from "../components/DataManager";
import Modal from "../components/Modal";
import VisionHeader from "../components/VisionHeader";
import VisionFooter from "../components/VisionFooter";
import { createEmptyStream } from "../libs/webRTC_utility";
import { useRecoilState, useRecoilValue } from "recoil";
import {
  selectedRoiState,
  imageWidthState,
  imageHeightState,
} from "../components/states";

export default function VisionPage() {
  const [allowWebcam, setAllowWebcam] = useState(false);
  const [hasWebcam, setHasWebcam] = useState(false);
  const [webcamEnabled, setWebcamEnable] = useState(false);
  const [webcamPlaying, setWebcamPlaying] = useState(false);

  const [demoLoaded, setDemoLoaded] = useState(false);
  const [demo, setDemo] = useState(false);
  const [loadedCoco, setLoadedCoco] = useState(false);
  const [processing, setProcessing] = useState(true);

  const [webcamLoaded, setWebcamLoaded] = useState(false);

  const [peerId, setPeerID] = useState("");
  const peerRef = useRef(null);
  const remoteRef = useRef(null);
  const rtcOutputRef = useRef(null);
  const [WebRTCMode, setWebRTCMode] = useState(false);
  const [WebRTCLoaded, setWebRTCLoaded] = useState(false);
  const selectedRois = useRecoilValue(selectedRoiState);
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);
  const [isOpen, setIsOpen] = useState(false)
  
  function closeModal() 
  {
    setIsOpen(false)
  }

  function openModal() 
  {
    setIsOpen(true)
  }


  useEffect(() => {
    if (remoteRef.current != null) {
      if (selectedRois.length > 0) {
        remoteRef.current.send(selectedRois);
      }
    }
  }, [selectedRois]);

  useEffect(() => {
    //Init peerJS
    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      peerRef.current = newPeer;
      newPeer.on("open", (id) => {
        setPeerID(id);
      });
      newPeer.on("connection", function (conn) {
        console.log("new connection", conn.peer);
        remoteRef.current = conn;
      });
      newPeer.on("call", (call) => {
        let fakeStream = createEmptyStream();
        call.answer(fakeStream);
        console.log("im getting a call ");
        call.on("stream", (remoteStream) => {
          rtcOutputRef.current.srcObject = remoteStream;
          rtcOutputRef.current.play();
        });
      });
    };
    initPeerJS();
  }, []);

  const handleDisableDemo = () => {
    setDemo(false);
    setImageWidth(640);
    setImageHeight(480);
  };

  return (
    <DashboardLayout>
      <Head>
        <title>Parkerr: Vision</title>
      </Head>
      <Modal isOpen={isOpen} closeModal={closeModal}>
    <DataManger/>
    </Modal>



      <div className="flex flex-col">
        <div className="flex flex-col items-center justify-center gap-4 rounded-lg">
          <div className="relative grid w-full h-20 gap-2 text-2xl font-bold text-white rounded-md">
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

            {!demo && allowWebcam ? <div /> : ""}
          </div>

          <div className="flex-col hidden gap-4 md:flex md:flex-row md:justify-between">
            <Toolbar
              setProcessing={setProcessing}
              allowWebcam={allowWebcam}
              setAllowWebcam={setAllowWebcam}
              processing={processing}
              hasWebcam={hasWebcam}
              loadedCoco={loadedCoco}
            ></Toolbar>
            <div>
              {demo || webcamPlaying || WebRTCLoaded ? <DrawingCanvas /> : null}

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
              <VisionFooter />
            </div>
            <RoisFeed openModal={openModal} />
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
