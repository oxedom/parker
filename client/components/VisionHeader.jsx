import DisplayInfo from "./DisplayInfo";

import { imageWidthState } from "./states";
import { useRecoilValue } from "recoil";

const VisionHeader = ({
  setAllowWebcam,
  setWebRTCMode,
  demo,
  handleDisableDemo,
  WebRTCMode,
  webcamEnable,
}) => {
  const imageWidth = useRecoilValue(imageWidthState);
  const btnStyle = "bg-orange-600 border-2 rounded-xl m-2 text-xl p-4 shadow  shadow-black"
  const handleWebcamSource = () => {
    setWebRTCMode(false)
    setAllowWebcam(true)
  }
  const handleRTCSource = () => {
    setAllowWebcam(false)
    setWebRTCMode(true)

  }

  return (
    <nav className={`flex justify-around `}>
      {!WebRTCMode && !webcamEnable ? <div className="flex gap-2 items-center ">
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
          <DisplayInfo></DisplayInfo>
        </>
      )
      }




    </nav>
  );
};

export default VisionHeader;
