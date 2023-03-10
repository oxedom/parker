import { async } from "@firebase/util";
import { useRef, useState } from "react";
import DefaultLayout from "../layouts/DefaultLayout";
import { firebaseConfig } from "../config";
import firebase from 'firebase/app';
import 'firebase/firestore';
// Output is a page that allows a user to allow his Webcam, get a link and share it with the person who will use it for the stream
const Output = () => {

    const [imageWidth, setImageWidth] = useState(640)
    const [imageHeight, setImageHeight] = useState(480)
    const servers = {
        iceServers: [
          {
            urls: ['stun:stun1.l.google.com:19302', 'stun:stun2.l.google.com:19302'],
          },
        ],
        iceCandidatePoolSize: 10,
      };

    console.log(firebase);
    // if (!firebase.apps.length) {
    //     firebase.initializeApp(firebaseConfig);
    // }
    const pc = new RTCPeerConnection(servers);
    // let localStream = null;
    let remoteStream = null;
    const allowWebcamRef = useRef(null)
    const outputRef = useRef(null)
    const handleAllowWebcam = async () => 
    {
 
        let localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        console.log(localStream)
        remoteStream = new MediaStream();

        localStream.getTracks().forEach((track) => {
            pc.addTrack(track, localStream)
        })

        pc.ontrack = (event) => {
            event.streams[0].getTracks().forEach((track) => {
              remoteStream.addTrack(track);
            });
          };

          outputRef.current.srcObject = localStream;
      

    }

    const handleAnswerCall = async () =>
    {




    }


    return ( <DefaultLayout>
        <main>

        <video ref={outputRef}>

        </video>


        <button className="bg-blue-500 p-5" ref={allowWebcamRef} onClick={handleAllowWebcam}> Allow Webcam</button>
        </main>


 
    </DefaultLayout> );
}
 
export default Output;