import { useState, useRef, useEffect } from "react";
import { firebaseConfig } from "../../config";
import { initializeApp } from "firebase/app";
import {
  collection,
  getDocs,
  doc,
  setDoc,
  getDoc,
  getFirestore,
  addDoc,
} from "firebase/firestore";


const ReceiverRTC = ({recevierRef}) => {

  const app = initializeApp(firebaseConfig);
  const [stream,setStream] = useState(null)
  useEffect(() => {
    if(stream != null) 
    {
      recevierRef.current.srcObject = stream
    }
  
  }, [stream])


  const servers = {
    iceServers: [
      {
        urls: [
          "stun:stun1.l.google.com:19302",
          "stun:stun2.l.google.com:19302",
        ],
      },
    ],
    iceCandidatePoolSize: 10,
  };
  let pc;
  if (typeof window !== "undefined") {
    pc = new RTCPeerConnection(servers);
  }

  const [connectID, setConnectID] = useState("");

  const handleSubmit = async (e) => {
    if (connectID.length < 5) {
      return;
    }
    let data = {};
    
    const db = getFirestore(app);
    const roomRef = doc(db, "rooms", connectID);
    const roomSnapshot = await getDoc(roomRef);
    data = { ...roomSnapshot.data() };
    pc = new RTCPeerConnection(servers);



    pc.addEventListener("track", async (e) => {
     

      let remoteVideo = document.getElementById('remoteVideo')
      const track = e.streams[0]
      console.log(track);
      setStream(track)
      recevierRef.current.srcObject = track
      

    });


    await pc.setRemoteDescription(data.offer)

  };
  return (
    <div className="bg-yellow-500 p-10">
      <input
        onChange={(e) => {
          setConnectID(e.target.value);
        }}
        type="text"
      ></input>
      <button onClick={handleSubmit}> submit</button>
    </div>
  );
};

export default ReceiverRTC;
