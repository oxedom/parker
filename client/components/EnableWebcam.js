import { useEffect, useState } from "react";
import { imageWidthState, imageHeightState } from "../components/states";
import { useRecoilState, useRecoilValue } from "recoil";

const EnableWebcam = ({
    hasWebcam, setHasWebcam, 
    setWebcamEnable,
    
    reload
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

  function detectWebcam(callback) {
    let md = navigator.mediaDevices;
    if (!md || !md.enumerateDevices) return callback(false);
    md.enumerateDevices().then((devices) => {
      callback(devices.some((device) => "videoinput" === device.kind));
    });
  }
  useEffect(() => {
    setTimeout(() => {
      if (!hasWebcam) {
        setWarrning(true);
      }
    }, 3000);

    const intervalId = setInterval(() => {
      detectWebcam(async (hasWebcamBoolean) => {
        await setUserSettings()
        setHasWebcam(hasWebcamBoolean);
        setWebcamEnable(hasWebcamBoolean)
        setWarrning(!hasWebcamBoolean);
      });
    }, 1000);

    return () => {
      clearInterval(intervalId);
    };
  }, [reload]);



  return (
    <div className="flex justify-center items-center flex-1  bg-pink-200  ">
      {(warrning && !hasWebcam) && (
        <div className="absolute bottom-3/4 uppercase font-bold transition-all duration-200 ease-in cursor-default  gap-2 flex-col items-center opacity-85 bg-red-400 w-full flex  text-white border-2 border-black p-5 ">
          <span className="text-4xl"> Unable to detect webcam </span>
          <span className="text-2xl"> please check your settings </span>{" "}
        </div>
      )}
      <button
        className={` w-max-[500px] p-10 text-6xl  spin-slow
         text-black border-2 font-bold mb-[300px] border-slate-800 uppercase ${
           hasWebcam
             ? "bg-slate-100"
             : "bg-gray-300  text-gray-100 cursor-not-allowed"
         } `}
      >
      {" "}
      </button>
    </div>
  );
};

export default EnableWebcam;
