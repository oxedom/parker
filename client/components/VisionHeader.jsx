import DisplayInfo from "./DisplayInfo";
import QRCode from "qrcode";
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
      className={`flex justify-around items-center animate-fade bg-orangeFadeSides rounded-md  `}
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
        <div className="grid items-center grid-cols-3 gap-10">
          <div
            onClick={handleBack}
            className="w-32 text-center text-gray-700 duration-200 bg-gray-200 border border-white rounded place-self-center hover:scale-105"
          >
            Back
          </div>
          <DisplayInfo></DisplayInfo>
          <div className="flex items-center justify-center gap-2">
            <button
              alt="streaming Link"
              className={`bg-orange-600 ${btnStyle} `}
              onClick={handleCopy}
            >
              Copy Link
            </button>
            <p className="text center "> OR </p>
            <Image
              width={80}
              alt="qr"
              quality={100}
              className="hover:scale-[2] duration-200"
              height={75}
              src={qrCodeURL}
            />
          </div>
        </div>
      )}
    </nav>
  );
};

export default VisionHeader;
