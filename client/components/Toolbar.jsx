import {
  imageHeightState,
  detectionThresholdState,
  thresholdIouState,
  autoDetectState,
  fpsState,
  showDetectionsState,
  vehicleOnlyState,
} from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import Modal from "./Modal";
import { useEffect, useState } from "react";
import Button from "./Button";
import ToogleSwitch from "./ToogleSwitch";

const Toolbar = ({
  processing,
  setProcessing,
  
  isModalOpen,
  allowWebcam,
  setAllowWebcam,
  loadedCoco,
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
  const [vehicleOnly, setVehicleOnly] = useRecoilState(vehicleOnlyState);
  const [detectionThreshold, setDetectonThreshold] = useRecoilState(
    detectionThresholdState
  );
  const [showDetections, setShowDetections] =
    useRecoilState(showDetectionsState);
  const [settingChange, setSettingsChange] = useState(false);
  const [iouThreshold, setIouThreshold] = useRecoilState(thresholdIouState);
  const [fps, setFps] = useRecoilState(fpsState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const [localDetectionThreshold, setLocalDetectionThreshold] =
    useState(undefined);
  const [localVehicleOnly, setLocalVehicleOnly] = useState(undefined);
  const [localIouThreshold, setLocalIouThreshold] = useState(undefined);
  const [localFps, setLocalFps] = useState(undefined);

  let sliderStyle = "flex mx-2 flex-col text-white justify-center  items-center"
  let detectInfo = `Detection Threshold: The minimum score that a vehicle detections is to be classifed as valid, recommended to be 50`;
  let iouInfo =
    "Advanced setting: Non-maximum Suppression threshold, recommended to between 50-75 ";
  let fpsInfo = `Render in N many secounds (The lower the faster and more compute demanding) 
  recommended to be 1 render per secound. 
  `;

  useEffect(() => {
    setSettingsChange(false);
    setLocalFps(fps);
    setLocalIouThreshold(iouThreshold * 100);
    setLocalDetectionThreshold(detectionThreshold * 100);
    setLocalVehicleOnly(vehicleOnly);
  }, [isModalOpen]);

  const handleAutoDetect = () => {
    if (autoDetect) {
      setAutoDetect(false);
    } else {
      setAutoDetect(true);
    }
  };

  const handleSaveSettings = () => {

    if (!settingChange) {
      return;
    }

    setDetectonThreshold(localDetectionThreshold / 100);
    setIouThreshold(localIouThreshold / 100);
    setFps(localFps);
    setVehicleOnly(localVehicleOnly);
    setProcessing(false);
    setSettingsChange(false);

    setTimeout(() => {
      setProcessing(true);
    }, 10);
  };

  function handleProcessing() {
    if (showDetections) {
      setShowDetections(false);
    }
    processing ? setProcessing(false) : setProcessing(true);
  }
  function handleWebcamToogle() {
    allowWebcam ? setAllowWebcam(false) : setAllowWebcam(true);
  }

  function handleDetectionsEnable() {
    if (processing && loadedCoco) {
      showDetections ? setShowDetections(false) : setShowDetections(true);
    }
  }

  function handleVehicleOnly() {
    setSettingsChange(true);

    vehicleOnly ? setLocalVehicleOnly(false) : setLocalVehicleOnly(true);
  }

  return (
    <div
      className={`md:w-[200px]  flex justify-between rounded-xl p-2  flex-col min-h-[${imageHeight}px]  bg-orange-500  `}
    >


      <div className="flex justify-center flex-col items-center gap-2  ">


        <div className={sliderStyle}>
          <label> Detection Threshold</label>
          <div className="grid grid-cols-2  ">
            <span> {localDetectionThreshold}% </span>
            <input
              type="range"
              min="10"
              max="100"
              className="mr-4"
              value={localDetectionThreshold}
              onChange={(e) => {
                setLocalDetectionThreshold(e.target.value);
                setSettingsChange(true);
              }}
            />
          </div>
        </div>

        <div className={sliderStyle}>
          <label> IOU Threshold</label>
          <div className="grid grid-cols-2 items-center  ">
            <span> {localIouThreshold}% </span>
            <input
              type="range"
              min="10"
              max="100"
              className="mr-4"
              label="NMS IOU Threshold"
              value={localIouThreshold}
              onChange={(e) => {
                setLocalIouThreshold(e.target.value);
                setSettingsChange(true);
              }}
            />
          </div>
        </div>

        <div className={sliderStyle}>
        <label> Render Rate</label>
          <div className="grid grid-cols-2 items-center    ">
          <span> {Math.floor(localFps * 100) / 100} FPS </span>
          <input
                    type="range"
                    min={0.0001}
                    step={0.1}
                    max={2}
                    className="mr-4"
                    label="Render rate"
                    value={localFps}
                    onChange={(e) => {
                      setLocalFps(e.target.value);
                      setSettingsChange(true);
                    }}
                  />
          </div>
        </div>


        <button    className={`text-white rounded-lg shadow outline outline-slate-700   ${settingChange ? "bg-blue-600 hover:bg-blue-700 text-white" : "bg-blue-400 text-slate-800 cursor-default " } bg-blue-500 outline-2 outline-black p-3`}       onClick={handleSaveSettings}> Save Settings</button>

 


        <ToogleSwitch
                text={"Webcam"}
                boolean={allowWebcam}
                callback={handleWebcamToogle}
              />

              <ToogleSwitch
                text={"Processing "}
                boolean={processing}
                callback={handleProcessing}
              />

              <ToogleSwitch
                text={"Vehicle Only "}
                boolean={localVehicleOnly}
                callback={handleVehicleOnly}
              />

            <ToogleSwitch
                text={"Draw Detections "}
                boolean={showDetections}
                callback={handleDetectionsEnable}
              />


      </div>

      <Button callback={handleAutoDetect} text="Auto detect" />


    </div>
  );
};

export default Toolbar;
