import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import ReceiverRTC from "../components/WebRTC/RecevierRTC";

const Input = () => {



  const [peer, setPeer] = useState(null);
  const [remotePeerIdValue, setRemotePeerIdValue] = useState('');
  const [peerId, setPeerId] = useState("")
  const peerRef = useRef(null)
  const streamRef = useRef(null)




  useEffect(() => {


    const initPeerJS = async () => {
      await   shareVideo()
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();

      peer.on("open", () => {

        console.log(`Input Peer connection open with ID: ${peer.id}`);
        setPeerId(peer.id);
        
      });

     
      

      peer.on('call', (call) => {
  
        call.answer(streamRef.current)
        console.log('im getting a call ');
        call.on("stream", (remoteSteam) => 
        {
          document.getElementById('video').srcObject = remoteSteam
        })




        
      })



    };
  
    initPeerJS();
  }, []);




  const shareVideo = async () => {
    try {
      const videostream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = videostream




    } catch (error) {
      console.error(error);
    }
  };




  return (
    <DefaultLayout>
    <div className="bg-green">
      <p>Current Peer ID: {peerId}</p>

      <video id="input" width="640" height="480" autoPlay></video>

    </div>


    <ReceiverRTC theID={peerId}> </ReceiverRTC>


    </DefaultLayout>
  );
};

export default Input;
