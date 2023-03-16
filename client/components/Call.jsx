import { useEffect, useRef } from "react";

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
        let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    
        const call = peerRef.current.call(peerID, stream)
    
    
        call.on('stream', (remoteStream) => {
          remoteVideoRef.current.srcObject = remoteStream
          remoteVideoRef.current.play();
        });
    
        
      }
    


    return ( <div onClick={handleConnect} className="bg-green-500 p-1 rounded">

        <button> Call </button>
    </div>  );
}
 
export default Call;