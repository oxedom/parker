import { useEffect, useRef } from "react";
import { createEmptyStream } from "../libs/webRTC_utility";
const Call = ({peerId, remoteVideoRef}) => {

    const peerRef = useRef(null)    

    function handleConnect (){
  
        call(peerId)
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

    async function call(peerID) 
      {
        let emptyStream = createEmptyStream()
        let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    
        const call = peerRef.current.call(peerID, emptyStream)
    
    
        call.on('stream', async (remoteStream) => {
          remoteVideoRef.current.srcObject = remoteStream
          remoteVideoRef.current.play();
        

        });
    
        
      }
    


    return ( <div onClick={handleConnect} className="bg-green-500 p-1 rounded">

        <button> Call </button>
    </div>  );
}
 
export default Call;