
import { useState, useRef, useEffect } from "react";


const ReceiverRTC = ({theID  }) => {

  const [remotePeerIdValue, setRemotePeerIdValue] = useState('');
  const [peerId, setPeerID] = useState("")
  const peerRef = useRef(null)
  const remoteVideoRef = useRef(null)

  function handleConnect (){
  
    call(peerId)
  }

  async function call(peerID) 
  {
    let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });

    const call = peerRef.current.call(peerID, stream)


    call.on('stream', (remoteStream) => {
      remoteVideoRef.current.srcObject = remoteStream
      remoteVideoRef.current.play();
    });

    
  }

  useEffect(() => {

    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      peerRef.current = newPeer
      console.log(peerRef.current);
    };

    initPeerJS();
  }, []);


  return (
    <div className="bg-yellow-500 p-10">

    <video id="video" ref={remoteVideoRef} width="640" height="480" autoPlay></video>

  
    <button className="w-10 bg-green-500 h-10" onClick={handleConnect}>Connect</button>
      <input value={peerId} onChange={(e) => {setPeerID(e.target.value)} }></input>
  </div>
  );
};

export default ReceiverRTC;
