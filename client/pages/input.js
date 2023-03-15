import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import ReceiverRTC from "../components/WebRTC/RecevierRTC";

const Input = () => {


  const [peer, setPeer] = useState(null);
  const [stream, setStream] = useState(null);
  const [peerId, setPeerId] = useState(null);




  useEffect(() => {
    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      
      newPeer.on("open", () => {
        console.log(`Input Peer connection open with ID: ${newPeer.id}`);
        setPeerId(newPeer.id);
        
      });

      newPeer.on('connection', function(conn) {
        conn.on('data', (data) => {

          console.log(data); // should log "hello world"
        });

        conn.send(stream)

      })

  



    };

    initPeerJS();
  }, []);




  const shareVideo = async () => {
    try {
      const videostream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(videostream);
      const video = document.querySelector("#video");
      video.srcObject = stream;
      video.play();




    } catch (error) {
      console.error(error);
    }
  };


  useEffect(() => {
    if(stream && peer) 
    {
      peer.on("connection", (conn) => {
        conn.send("yellow");
    })
    }


  }, [stream])

  return (
    <DefaultLayout>
    <div className="bg-green">
      <p>Current Peer ID: {peerId}</p>
      <button className="bg-green-500" onClick={shareVideo}>Share Video</button>
      <video id="video" width="640" height="480" autoPlay></video>

    </div>


    <ReceiverRTC theID={peerId}> </ReceiverRTC>


    </DefaultLayout>
  );
};

export default Input;
