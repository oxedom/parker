import { useEffect, useState } from "react";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";


const EnableWebcam = ({
    hasWebcam, setHasWebcam, 
    setWebcamEnable,
    reload,
    setReload
 }) => {

  const [warrning, setWarrning] = useState(false);
  const [imageWidth, setImageWidth] = useRecoilState(imageWidthState);
  const [imageHeight, setImageHeight] = useRecoilState(imageHeightState);

  const getSetting = async () => {
    let stream = await navigator.mediaDevices.getUserMedia({ video: true });
    let { width, height } = stream.getTracks()[0].getSettings();
    return { width, height };
  };

  async function setUserSettings () {
    let {width,height} = await getSetting();
    setImageWidth(width)
    setImageHeight(height)
    
 
 
  }

  function handleReload() 
  {
    setReload((prev) => { return prev+1})
   }

  function detectWebcam(callback) {
    let md = navigator.mediaDevices;
    if (!md || !md.enumerateDevices) return callback(false);
    md.enumerateDevices().then((devices) => {
      callback(devices.some((device) => "videoinput" === device.kind));
    });
  }
  useEffect(() => {
    
    const intervalId = setInterval(() => {

      detectWebcam(async (hasWebcamBoolean) => {
        if(hasWebcamBoolean) 
        {
          await setUserSettings()
          setHasWebcam(hasWebcamBoolean);
          setWebcamEnable(hasWebcamBoolean)
          setWarrning(!hasWebcamBoolean);
        }
        else 
        {
          setHasWebcam(false)
          setWarrning(true)
        }
    });
    }, 1000);

    return () => {
      clearInterval(intervalId);
    };
  }, [hasWebcam]);



  return (
    <div className="flex justify-center items-center flex-1  bg-pink-200  ">
      {(warrning && !hasWebcam) && (
        <div className="absolute bottom-2/4 uppercase font-bold transition-all duration-200 ease-in cursor-default  gap-2 flex-col items-center opacity-85 bg-red-400 w-full flex  text-white border-2 border-black p-5 ">
          <span className="text-4xl"> Unable to detect webcam </span>
          <span className="text-2xl"> please check your settings </span>{" "}
          <button onClick={handleReload} className="p-5 bg-gray-600 border-black border-2 uppercase"> Reload Webcam</button>
        </div>
      )}


      {/* <img
        className={` w-max-[500px] animate-spin p-10 text-6xl  spin-slow
         text-black border-2 font-bold mb-[300px] border-slate-800 uppercase ${
           hasWebcam
             ? "bg-slate-100"
             : "bg-gray-300  text-gray-100 cursor-not-allowed"
         } `}
      /> */}
      {" "}
 
    </div>
  );
};

export default EnableWebcam;
