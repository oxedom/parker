import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { firebaseConfig } from "../config";
import { initializeApp } from "firebase/app";
import {
  collection,
  getDocs,
  doc,
  setDoc,
  getFirestore,
  addDoc,
} from "firebase/firestore";
import ReceiverRTC from "../components/WebRTC/RecevierRTC";

// import uniqid from uniqid

const Output = () => {
  const [database, setDatabase] = useState(null);
  const [connectUser, setConnectedUser] = useState(null);
  const [localStream, setLocalStream] = useState(null);
  const [localConnection, setLocalConnection] = useState(null);
  const [offerID, setOfferID] = useState(null);
  const allowBtnRef = useRef(null);
  const offerBtnRef = useRef(null);
  const outputRef = useRef(null);
  const recevierRef = useRef(null);
  const app = initializeApp(firebaseConfig);

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

  const handleOffer = async () => {
    const db = getFirestore(app);
    // const callDoc = doc(db, 'calls')
    let _localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    outputRef.current.srcObject = _localStream;

    console.log(pc);
    _localStream.getTracks().forEach((track) => {
      console.log(track);
      // pc.addTrack(track)
      pc.addTrack(track, _localStream);
    });
    // pc.addStream(_localStream)
    pc.ontrack = (event) => {
      event.streams[0].getTracks().forEach((track) => {
        remoteStream.addTrack(track);
      });
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const roomWithOffer = {
      offer: {
        type: offer.type,
        sdp: offer.sdp,
      },
      candidates: [],
    };

    pc.addEventListener("icecandidate", async (e) => {
      if (e.candidate) {
        const json = e.candidate.toJSON();
        roomWithOffer.candidates.push(json);
        const roomRef = await addDoc(collection(db, "rooms"), {
          ...roomWithOffer,
        });
        const roomId = roomRef.id;
        setOfferID(roomId);
      }

      // console.log(e);
      // console.log(e.canidate.toJSON());
    });

    // const roomRef = await setDoc(callDoc, roomWithOffer)

    // const roomId = roomRef.id;

    // setOfferID(roomId);
  };

  const handleAnswerCall = async () => {};

  return (
    <DefaultLayout>
      <main>
        <video
          className=""
          autoPlay={true}
          muted={true}
          ref={outputRef}
        ></video>

        <video ref={recevierRef} muted={true} autoPlay={true}></video>

        <div>
          <p> {offerID} </p>
          <button onClick={handleOffer} className="bg-blue-500 p-5">
            {" "}
            create Offer
          </button>
        </div>

        <ReceiverRTC recevierRef={recevierRef}></ReceiverRTC>
      </main>
    </DefaultLayout>
  );
};

export default Output;
