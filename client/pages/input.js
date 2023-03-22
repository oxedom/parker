import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { useRouter } from "next/router";

const Input = () => {

  const [remoteID, setRemoteID] = useState("")
  const inputRef = useRef(null)
  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const router = useRouter()



  useEffect(() => {



    const initPeerJS = async () => {
      
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();
      peerRef.current = peer
    };

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

  const call = async () => 
  {
    const videoStream = await shareVideo();
    console.log(videoStream);
    console.log(videoStream.getVideoTracks()[0].getSettings());
    const call = peerRef.current.call(router.query.remoteID, videoStream);

    call.on("stream", async (remoteStream) => {
      console.log(remoteStream);
    });

  }

  return (
    <DefaultLayout>
      <div className="bg-green-500 w-[250px]">
        <p>Current Peer ID: {remoteID}</p>
        <video autoPlay={true} ref={inputRef}></video>
        <button className="bg-yellow-500 p-5" onClick={call}>
          {" "}
          Call
        </button>
      </div>
    </DefaultLayout>
  );
};

export default Input;
