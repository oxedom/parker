import DisplayInfo from "./DisplayInfo";
import QRCode from "qrcode";
import { useEffect, useState } from "react";

import Button from "./Button";
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
  const btnStyle = `border rounded-xl m-2 text-xl p-2 shadow-sm shadow-black text-center   hover:scale-105 duration-200 ${!allowWebcam && !WebRTCMode && !demo ? " animate-pulse" : ""
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
    const currentDomain = new URL(window.location.href);
    currentDomain.pathname = "reroute";
    currentDomain.searchParams.set("remoteID", peerId);

    generateQR(currentDomain.href);
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
    const currentDomain = new URL(window.location.href);
    currentDomain.pathname = "reroute";
    currentDomain.searchParams.set("remoteID", peerId);
    navigator.clipboard.writeText(currentDomain.href);
  };

  return (
    <nav
      className={`flex justify-between items-center animate-fade bg-black/60 backdrop-blur-sm rounded-md  `}
    >
      {!WebRTCMode && !allowWebcam && !demo ? (
        <div className="flex flex-row items-center justify-between w-full ml-3 mr-6">
          <section>
            <button
              onClick={handleDemoSource}
              className={`bg-purple-600 ${btnStyle} `}
            >
              Video Demo
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
          </section>
        </div>
      ) : (
        <div className="flex items-center justify-between w-full grid-cols-3 gap-10 px-3">
          <Button intent="destructive" onClick={handleBack}>
            Back
          </Button>
          <DisplayInfo></DisplayInfo>
          <section className="relative group">
            <Button alt="streaming Link" onClick={handleCopy}>
              Copy Link
            </Button>
            <Image
              width={80}
              alt="qr"
              quality={100}
              className="scale-[2] hidden group-hover:block duration-200 absolute bottom-[5rem] left-1/2 translate-x-[-50%]"
              height={75}
              src={qrCodeURL}
            />
          </section>
        </div>
      )}
    </nav>
  );
};

export default VisionHeader;
