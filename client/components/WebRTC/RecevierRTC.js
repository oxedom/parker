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


const ReceiverRTC = () => {
  const kekRef = useRef(null);
  const app = initializeApp(firebaseConfig);
  const [videoStream, setStream] = useState(null)

  // useEffect(() => {
  //   if(videoStream != null) 
  //   {
  //     // const videoTracks = videoStream.getVideoTracks();

  //     // console.log(videoStream);
  //     kekRef.current.srcObject = videoStream
   
  //   }

  // }, [videoStream])

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
    const { offer } = data;
    console.log(offer);

    
    await pc.setRemoteDescription(offer);
    // const answer = await pc.createAnswer();
    // await pc.setLocalDescription(answer);
    console.log(    pc.ondatachannel);

    pc.addEventListener("track", (e) => {
      console.log(e);
      alert("my name is track")
      const videoElement = document.getElementById('output')
      videoElement.srcObject = e.streams[0];
    });
  
    

  };
  return (
    <div className="bg-yellow-500 p-10">
      <video id="output"  autoPlay={true} className="p-1"  width={640} height={480} key="1000000000000"  ref={kekRef}/>
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
