import DisplayInfo from "./DisplayInfo";
import QRCode from 'qrcode'
import { imageWidthState } from "./states";
import { useRecoilValue } from "recoil";
import { useEffect, useState, useRef } from "react";
import Image from "next/image";
const VisionHeader = ({
  setAllowWebcam,
  peerId,
  setWebRTCMode,
  demo,
  handleDisableDemo,
  WebRTCMode,
  allowWebcam,
}) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const [qrCodeURL, setQRcodeURL] = useState("")
  const btnStyle = "bg-orange-600 border-2 rounded-xl m-2 text-xl p-4 shadow  shadow-black"
  const handleWebcamSource = () => {
    setWebRTCMode(false)
    setAllowWebcam(true)
  }
  const handleRTCSource = () => {
    setAllowWebcam(false)
    setWebRTCMode(true)

  }

  const generateQR = async text => {
    try {
     let qrString = await QRCode.toDataURL(text)
      setQRcodeURL(qrString)
    } catch (err) {
      console.error(err)
    }
  }


  useEffect(()=> {
   generateQR(`https://www.sam-brink.com/input?remoteID=${peerId}`)

  }, [peerId])



  const handleCopy = () => 
  {
    navigator.clipboard.writeText(`https://www.sam-brink.com/input?remoteID=${peerId}`);
  }

  return (
    <nav className={`flex justify-around `}>
   
      {!WebRTCMode && !allowWebcam ? <div className="flex gap-2 items-center ">
        <button onClick={handleWebcamSource} className={btnStyle}> Webcam Video Source</button>
         <strong className="text-center text-3xl ">   </strong>
        <button  onClick={handleRTCSource} className={btnStyle}> Remote Video Source </button>
      </div> : 
      
      demo ? (
        <div onClick={handleDisableDemo}>
          <h5 className="font-bold p-4   text-gray-200 justify-self-start hover:text-white bg-orange-600   text-2xl  duration-300  ">
            {" "}
            Exit{" "}
          </h5>
          <span className=""> </span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3  items-center gap-10"  >
          <div></div>
          <DisplayInfo></DisplayInfo>
          <div className="flex gap-2 items-center">
            {WebRTCMode ?   <button alt="streaming Link"  className={btnStyle}  onClick={handleCopy}> Copy Link </button> : ""}
            <p className="text center "> OR </p>
            <Image  width={75} alt="qr" quality={100}   height={75} src={qrCodeURL} />
          </div>
          </div>

        
        </>
      )
      }




    </nav>
  );
};

export default VisionHeader;
