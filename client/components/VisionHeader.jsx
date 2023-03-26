import DisplayInfo from "./DisplayInfo";
import QRCode from 'qrcode'
import { imageWidthState } from "./states";
import { useRecoilValue } from "recoil";
import { useEffect, useState } from "react";
import Image from "next/image";
const VisionHeader = ({
  setAllowWebcam,
  peerId,
  setWebRTCMode,
  demo,
  WebRTCMode,
  allowWebcam,
  setWebcamLoaded,
  webcamLoaded,
  setDemo,
}) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const [qrCodeURL, setQRcodeURL] = useState("")
  const btnStyle = "bg-orange-600 border-2 rounded-xl m-2 text-xl p-2 shadow  shadow-black text-center"
  const handleWebcamSource = () => {
    setWebRTCMode(false)
    setDemo(false)
    setAllowWebcam(true)

  }
  const handleRTCSource = () => {
    setAllowWebcam(false)
    setDemo(false)
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


  const handleBack = () => 
  {
    setAllowWebcam(false)
    setWebRTCMode(false)
    setWebcamLoaded(false)
    setDemo(false)
  }

  const handleDemoSource = () => {
    setDemo(true)
    setWebcamLoaded(false)
    setAllowWebcam(false)
    setWebRTCMode(false)
   
  }


  const handleCopy = () => 
  {
    navigator.clipboard.writeText(`https://www.sam-brink.com/input?remoteID=${peerId}`);
  }

  return (
    <nav className={`flex justify-around items-center  `}>
   
      {!WebRTCMode && !allowWebcam && !demo ? <div className="flex gap-2 items-center ">

      <button  onClick={handleDemoSource} className={btnStyle}> Demo  </button>
        <button onClick={handleWebcamSource} className={btnStyle}> Webcam Video Source</button>
        <button  onClick={handleRTCSource} className={btnStyle}> Remote Video Source </button>

      </div> : 
      
   
        
          <div className="grid grid-cols-3  items-center gap-10"  >
          <div onClick={handleBack} > Back  </div>
          <DisplayInfo></DisplayInfo>
          <div className="flex gap-2 items-center">
            {WebRTCMode ?   <>
              <button alt="streaming Link"  className={btnStyle}  onClick={handleCopy}> Copy Link </button> 
              <p className="text center "> OR </p>
            <Image  width={75} alt="qr" quality={100}   className="hover:scale-[2] duration-200"  height={75} src={qrCodeURL} />
            </>: ""}
      
          </div>
          </div>

        
      
      
      }




    </nav>
  );
};

export default VisionHeader;
