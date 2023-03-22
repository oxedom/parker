import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { useRouter } from "next/router";
import NoSleep from 'nosleep.js';

const Input = () => {

  const inputRef = useRef(null)
  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const callRef = useRef(null)
  const router = useRouter()
  const noSleepRef = useRef(null)
  const [connection, setConnection] = useState(false)
  

  


  useEffect(() => {

    const initNoSleep = async () => 
    {
      const { default: NoSleep } = await import("nosleep.js");
      const noSleep = new NoSleep();  
      try {
        noSleep.enable()
      } catch (error) {
        console.log(error);
      }
    
    }
    
    const initPeerJS = async () => {
      
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();
      peerRef.current = peer
    };
    initNoSleep()
    initPeerJS();


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
      setConnection(true)
    });

  }

  return (
    <DefaultLayout>
      <div className="bg-filler h-full flex gap-5 flex-col justify-center items-center">
        {   inputRef.current === null ? <p className="text-2xl text-white py-2">  Please call </p> : null}
        <video autoPlay={true} className="hidden sm:block rounded-xl" ref={inputRef}></video>
        <p className="text-5xl py-2 text-white ">  Connection: {connection ? "Established" : "Pending"} </p>
        
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
