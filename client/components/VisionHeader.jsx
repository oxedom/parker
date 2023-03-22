import DisplayInfo from "./DisplayInfo";

import { imageWidthState } from "./states";
import { useRecoilValue } from "recoil";

const VisionHeader = ({
  setWebRTCMode,
  setAllowWebcam,
  demo,
  handleDisableDemo,
  WebRTCMode,
  webcamEnable,
}) => {
  const imageWidth = useRecoilValue(imageWidthState);

  return (
    <nav className={`flex justify-around `}>
      {demo ? (
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
      )}
    </nav>
  );
};

export default VisionHeader;
