import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { useRouter } from "next/router";


const Input = () => {

  const inputRef = useRef(null)
  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const callRef = useRef(null)
  const router = useRouter()

  function updateScreen() {
    // update screen here
    window.requestAnimationFrame(updateScreen);
  }
  

  


  useEffect(() => {


    let keepAwakeID = setInterval(updateScreen, 1000);
    const initPeerJS = async () => {
      
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();
      peerRef.current = peer
    };

    initPeerJS();
    return () => { clearInterval(keepAwakeID)}
  }, []);

  const shareVideo = async () => {
    try {
      const videostream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { exact: "environment" } },
      });

      streamRef.current = videostream;
      inputRef.current.srcObject = videostream;
      return videostream
    } catch (error) {
      try {
        const videostream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });

        streamRef.current = videostream;
        inputRef.current.srcObject = videostream;
        return videostream
      } catch (error) {
        console.error(error);
      }
    }
  };


  const hangup = () => 
  {
    //  callRef.current.close()
  }

  const call = async () => 
  {
   

    const videoStream = await shareVideo();

    const call = peerRef.current.call(router.query.remoteID, videoStream);
    callRef.current = call
    call.on("stream", async (remoteStream) => {
    });

  }

  return (
    <DefaultLayout>
      <div className="bg-filler h-full flex gap-5 flex-col justify-center items-center">
        {   inputRef.current === null ? <p className="text-2xl text-white py-2">  Please call </p> : null}
        <video autoPlay={true} className="rounded-xl" ref={inputRef}></video>
        
        <div className="flex flex-col md:flex-row gap-4">
        <button className="bg-green-400 py-2 rounded-lg shadow-sm active:bg-green-500 hover:bg-green-500 text-white  font-bold text-4xl p-5 w-[250px]" onClick={call}>
          {" "}
          Call
        </button>
        <button className="bg-red-400 py-2 rounded-lg shadow-sm active:bg-green-500 hover:bg-red-500 text-white  font-bold text-4xl p-5 w-[250px]" onClick={hangup}>
          {" "}
          Hang up
        </button>
        </div>

      </div>
    </DefaultLayout>
  );
};

export default Input;
