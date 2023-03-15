import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { firebaseConfig } from "../config";
import { initializeApp } from "firebase/app";
import dynamic from 'next/dynamic';
import {
  collection,
  getDocs,
  doc,
  setDoc,
  getFirestore,
  addDoc,
} from "firebase/firestore";


// import { Peer } from "peerjs";

// import uniqid from uniqid



const Input = () => {
  const [database, setDatabase] = useState(null);
  const [connectUser, setConnectedUser] = useState(null);
  const [localStream, setLocalStream] = useState(null);
  const [localConnection, setLocalConnection] = useState(null);
  const [offerID, setOfferID] = useState(null);
  const allowBtnRef = useRef(null);
  const offerBtnRef = useRef(null);
  const outputRef = useRef(null);
  const recevierRef = useRef(null);


  const [peer, setPeer] = useState(null);
  const [stream, setStream] = useState(null);
  const [peerId, setPeerId] = useState("");


    useEffect(() =>
    {
    
      import('peerjs').then(({ default: Peer }) => {
   
       const newPeer = new Peer(peerId);
          newPeer.on('open', () => {
            console.log(`Peer connection open with ID ${newPeer.id}`);
            setPeerId(newPeer.id)
            setPeer(newPeer);
          });

            // Set up the event listener for when someone else tries to connect to this peer
            newPeer.on('connection', (conn) => {
              console.log(`New connection from ${conn.peer}`);
            });

          });
    }, [])

  const handleOffer = async () => {

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        setStream(stream);
        outputRef.current.srcObject = stream;
      })
      .catch((error) => {
        console.error('Error accessing user media:', error);
      });

      return () => {
        newPeer.destroy();

      };

  };

  return (
    <DefaultLayout>
      <main>
        <div className="bg-green-500 p-5">
          <p> {peerId}</p>
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
        </div>
        <div className="bg-yellow-500 p-5">
        {/* <ReceiverRTC recevierRef={recevierRef}></ReceiverRTC> */}
        </div>
        


      </main>
    </DefaultLayout>
  );
};

export default Input;
