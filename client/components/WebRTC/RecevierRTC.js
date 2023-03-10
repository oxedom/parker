import { useState , useRef} from "react";
import { firebaseConfig} from "../../config"
import  {initializeApp} from 'firebase/app'
import { collection, getDocs , doc, setDoc, getDoc, getFirestore, addDoc} from 'firebase/firestore';

const ReceiverRTC = () => {

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
    let pc
    if (typeof window !== 'undefined') { 
        pc = new RTCPeerConnection(servers);
     
      }

    const [connectID, setConnectID] = useState("")


      

    const handleSubmit = async (e) => 
    {
        if((connectID.length < 5)) {return}
        let data = {}        
        const db = getFirestore(app)
        const roomRef = doc(db, "rooms", connectID)
        const roomSnapshot = await getDoc(roomRef)
        data = {...roomSnapshot.data()}
        pc = new RTCPeerConnection(servers);
        const {offer} = data;

        pc.addEventListener(('track'), (e) => 
        {
        outputRef.current.srcObject = e.streams[0]
        })
        await pc.setRemoteDescription(offer)
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer)
        const streams = await pc.getRemoteStreams()
        // console.log(pc);
        // outputRef.current.srcObject = streams[0]
    }
    return ( <div className="bg-yellow-500 p-10">

  
            <video autoPlay={true} muted={true}  ref={outputRef}></video>
            <input onChange={(e) => {setConnectID(e.target.value)}} type='text'></input>
            <button onClick={handleSubmit}> submit</button>
     
    </div> );
}
 
export default ReceiverRTC;