
import { useState, useRef, useEffect } from "react";


const ReceiverRTC = ({theID  }) => {
  const [peer, setPeer] = useState(null);
  const [stream, setStream] = useState(null);
  const [peerId, setPeerID] = useState("")

  function handleConnect (){
    alert()
    var conn = peer.connect(peerId);
    console.log(conn);
    conn.on('open', () => {
      conn.send('hello world');
    });

    conn.on('data', (data) => {

      console.log(data); // should log "hello world"
    });

  }

  useEffect(() => {
    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const newPeer = new Peer();
      setPeer(newPeer)
      // newPeer.on('connection', function(conn) { console.log(conn)});
    };

    initPeerJS();
  }, []);


  return (
    <div className="bg-yellow-500 p-10">
    {/* <video id="video" width="640" height="480" autoPlay></video> */}

  
    <button className="w-10 bg-green-500 h-10" onClick={handleConnect}>Connect</button>
      <input value={peerId} onChange={(e) => {setPeerID(e.target.value)} }></input>
  </div>
  );
};

export default ReceiverRTC;
