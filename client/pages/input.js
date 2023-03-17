import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";


const Input = () => {

  const [peer, setPeer] = useState(null);
  const [remotePeerIdValue, setRemotePeerIdValue] = useState('');
  const [peerId, setPeerId] = useState("")
  const inputRef = useRef(null)
  const streamRef = useRef(null)


  useEffect(() => {


    const initPeerJS = async () => {

      const { default: Peer } = await import("peerjs");
      const peer = new Peer();

      peer.on("open", () => {

        console.log(`Input Peer connection open with ID: ${peer.id}`);
        setPeerId(peer.id);
        
      });

      peer.on('call', (call) => {
        call.answer(streamRef.current)
        console.log('im getting a call ');
      })

    };
  
    initPeerJS();
  }, []);




  const shareVideo = async () => {
    try {
      const videostream = await navigator.mediaDevices.getUserMedia({ video:
        {    facingMode: { exact: "environment" },} })

      streamRef.current = videostream
      inputRef.current.srcObject = videostream
    } catch (error) {

      try {
        const videostream = await navigator.mediaDevices.getUserMedia({ video: true } )
        
  
        streamRef.current = videostream
        inputRef.current.srcObject = videostream
  
      } catch (error) {
        console.error(error);
      }


    }
  };


  return (
    <DefaultLayout>
    <div className="bg-green-500">
      <p>Current Peer ID: {peerId}</p>
      <video autoPlay={true} ref={inputRef}></video>
      <button onClick={shareVideo}></button>
    </div>
    </DefaultLayout>
  );
};

export default Input;
