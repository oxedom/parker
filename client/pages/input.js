import { useEffect, useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { firebaseConfig} from "../config";
import  {initializeApp} from 'firebase/app'
import { collection, getDocs , doc, setDoc, getFirestore, addDoc} from 'firebase/firestore';
import ReceiverRTC from "../components/WebRTC/RecevierRTC";
// import uniqid from uniqid



const Output = () => {

    const [database, setDatabase] = useState(null)
    const [connectUser, setConnectedUser] = useState(null)
    const [localStream, setLocalStream] = useState(null)
    const [localConnection, setLocalConnection] = useState(null)
    const [offerID, setOfferID] = useState(null)
    const allowBtnRef = useRef(null)
    const offerBtnRef = useRef(null)
    const outputRef = useRef(null)

  
    const app = initializeApp(firebaseConfig)
 

    const servers = {
        iceServers: [
          {
            urls: ['stun:stun1.l.google.com:19302', 'stun:stun2.l.google.com:19302'],
          },
        ],
        iceCandidatePoolSize: 10,
      };

      let pc;

      if (typeof window !== 'undefined') { 
        pc = new RTCPeerConnection(servers);

      }
      






    const handleOffer = async () => 
    {
      let _localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      outputRef.current.srcObject = _localStream;
      setLocalStream(_localStream)
      _localStream.getTracks().forEach((track) => {
          pc.addTrack(track, _localStream)
    
      })

      pc.ontrack = (event) => {
          event.streams[0].getTracks().forEach((track) => {
            remoteStream.addTrack(track);
          });
        };

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      const db = getFirestore(app)

      const roomWithOffer = {
        offer: {
            type: offer.type,
            sdp: offer.sdp
        }
    }
      // const docRef = doc(db, 'calls', "0");
      const roomRef =await addDoc(collection(db, "rooms"), {
        ...roomWithOffer
      });
      const roomId = roomRef.id;

      // const callDoc = collection('calls').doc();
      // console.log(callDoc);
      setOfferID(roomId)
   
      
    }
    const handleAnswerCall = async () =>
    {

    }


    return ( <DefaultLayout>
        <main>

        <video className="" autoPlay={true} muted={true} ref={outputRef}>

        </video>


     

      
      <div>
      <p> {offerID} </p>
      <button  onClick={handleOffer} className="bg-blue-500 p-5"> create Offer</button>
      </div>
     
     <ReceiverRTC></ReceiverRTC>
        </main>



 
    </DefaultLayout> );
}
 
export default Output;