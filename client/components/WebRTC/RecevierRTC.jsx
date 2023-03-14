import { useState, useRef, useEffect } from "react";
import { Peer } from "peerjs";

const ReceiverRTC = ({ recevierRef }) => {
  const [peer, setPeer] = useState(null);
  const [stream, setStream] = useState(null);
  const [peerId, setPeerId] = useState("");
  const videoRef = useRef(null);

  const [connectID, setConnectID] = useState("");

  useEffect(() => {
    // Create a new Peer instance
    const newPeer = new Peer(12345);

    // Set up the event listener for when the peer connection is open
    newPeer.on("open", () => {
      console.log(`Peer connection open with ID ${newPeer.id}`);
      setPeer(newPeer);
    });

    // Set up the event listener for when someone else tries to connect to this peer
    newPeer.on("connection", (conn) => {
      console.log(`New connection from ${conn.peer}`);
    });

    // Clean up the Peer instance and video stream when the component unmounts
    return () => {
      newPeer.destroy();
    };
  }, []);

  const handleConnect = () => {
    // Get access to the user's webcam and set it as the video stream
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        setStream(stream);
        videoRef.current.srcObject = stream;

        // Connect to the peer with the specified ID
        const conn = peer.connect(peerId);
        conn.on("open", () => {
          console.log(`Connected to peer ${peerId}`);
        });
        conn.on("data", (data) => {
          console.log(`Received data: ${data}`);
        });
        conn.on("close", () => {
          console.log(`Connection to peer ${peerId} closed`);
        });
      })
      .catch((error) => {
        console.error("Error accessing user media:", error);
      });
  };

  return (
    <div>
      <h1>Peer Viewer</h1>
      <label>
        Enter a Peer ID:
        <input
          type="text"
          value={peerId}
          onChange={(e) => setPeerId(e.target.value)}
        />
      </label>
      <button onClick={handleConnect}>Connect</button>
      <video ref={videoRef} autoPlay />
      {peer && <p>My Peer ID: {peer.id}</p>}
    </div>
  );
};

export default ReceiverRTC;
