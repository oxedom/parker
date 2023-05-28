import DisplayInfo from "./DisplayInfo";
import QRCode from "qrcode";
import { useEffect, useState } from "react";

import Button from "./Button";

const VisionHeader = ({
  setAllowWebcam,
  peerId,
  setWebRTCMode,
  demo,
  WebRTCMode,
  allowWebcam,
  setWebcamLoaded,
  handleDisableDemo,
  setDemo,
}) => {
  const [qrCodeURL, setQRcodeURL] = useState("");
  const btnStyle = `border rounded-xl m-2 text-xl p-2 shadow-sm shadow-black text-center   hover:scale-105 duration-200 ${
    !allowWebcam && !WebRTCMode && !demo ? " animate-pulse" : ""
  } hover:shadow-none`;
  const handleWebcamSource = () => {
    setWebRTCMode(false);
    setDemo(false);
    setAllowWebcam(true);
  };
  const handleRTCSource = () => {
    setAllowWebcam(false);
    setDemo(false);
    setAllowWebcam(false);
    setWebRTCMode(true);
  };

  const generateQR = async (text) => {
    try {
      let qrString = await QRCode.toDataURL(text);
      setQRcodeURL(qrString);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    generateQR(`https://www.parkerr.org/reroute?remoteID=${peerId}`);
  }, [peerId]);

  const handleBack = () => {
    setAllowWebcam(false);
    setWebRTCMode(false);
    setWebcamLoaded(false);
    handleDisableDemo();
  };

  const handleDemoSource = () => {
    setDemo(true);
    setWebcamLoaded(false);
    setAllowWebcam(false);
    setWebRTCMode(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(
      `https://www.parkerr.org/reroute?remoteID=${peerId}`
    );
  };

  return (
    <nav
      className={`flex justify-between items-center animate-fade bg-black/60 backdrop-blur-sm rounded-md  `}
    >
      {!WebRTCMode && !allowWebcam && !demo ? (
        <div className="grid items-center grid-cols-3 ">
          <button
            onClick={handleDemoSource}
            className={`bg-purple-600 ${btnStyle} `}
          >
            Demo
          </button>
          <button
            onClick={handleWebcamSource}
            className={`bg-purple-600 ${btnStyle} `}
          >
            Webcam
          </button>
          <button
            onClick={handleRTCSource}
            className={`bg-purple-600 ${btnStyle} `}
          >
            Remote
          </button>
        </div>
      ) : (
        <div className="flex items-center justify-between w-full grid-cols-3 gap-10 px-3">
          <Button intent="destructive" onClick={handleBack}>
            Back
          </Button>
          <DisplayInfo></DisplayInfo>
          <Button alt="streaming Link" onClick={handleCopy}>
            Copy Link
          </Button>
        </div>
      )}
    </nav>
  );
};

export default VisionHeader;
