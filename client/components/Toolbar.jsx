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

import { useEffect, useState } from "react";

import Slider from "./Slider";
import Button from "./Button";
import ToogleSwitch from "./ToogleSwitch";
import Accordion from "../components/Accordion";

const Toolbar = ({ processing, setProcessing, loadedCoco }) => {
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

  let detectInfo = `Detection Threshold: The minimum score that a vehicle detections is to be classifed as valid, recommended to be 50`;
  let iouInfo =
    "Advanced setting: Non-maximum Suppression threshold, recommended to between 50-75 ";
  let fpsInfo = `Render in N many secounds (The lower the faster and more compute demanding) 
  recommended to be 1 render per secound. 
  `;

  const labelStyle = "font-bold drop-shadow-sm text-left";
  const sliderStyle = "flex mx-2 flex-col text-white justify-center";
  useEffect(() => {
    setSettingsChange(false);
    setLocalFps(fps);
    setLocalIouThreshold(iouThreshold * 100);
    setLocalDetectionThreshold(detectionThreshold * 100);
    setLocalVehicleOnly(vehicleOnly);
  }, []);

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
    processing ? setProcessing(false) : setProcessing(true);
  }

  function handleDetectionsEnable() {
    if (processing && loadedCoco) {
      showDetections ? setShowDetections(false) : setShowDetections(true);
    }
  }

  function handleVehicleOnly() {
    setSettingsChange(true);

    localVehicleOnly ? setLocalVehicleOnly(false) : setLocalVehicleOnly(true);
  }

  return (
    <div
      className="md:w-[200px] flex justify-between rounded-xl flex-col bg-black/60 backdrop-blur-sm"
      style={{
        minHeight: imageHeight + "px",
      }}
    >
      <div className="flex flex-col justify-center gap-2 ">
        <Accordion imageHeight={imageHeight} title={"Settings"}>
          <div className="flex flex-col gap-3">
            <ToogleSwitch
              text="Processing"
              boolean={processing}
              callback={handleProcessing}
            />

            <ToogleSwitch
              text="Show Boxes"
              boolean={showDetections}
              callback={handleDetectionsEnable}
            />

            <ToogleSwitch
              text="Vehicle Only"
              boolean={localVehicleOnly}
              callback={handleVehicleOnly}
            />
          </div>

          <Slider
            state={[localDetectionThreshold, setLocalDetectionThreshold]}
            setSettingsChange={setSettingsChange}
            label="Detection Threshold"
          />

          <Slider
            state={[localIouThreshold, setLocalIouThreshold]}
            setSettingsChange={setSettingsChange}
            label="IOU Threshold"
          />

          <Slider
            state={[localFps, setLocalFps]}
            setSettingsChange={setSettingsChange}
            label="Render Rate"
            max={2}
            min={0.01}
            step={0.1}
            unit=" FPS"
          />

          <Button
            className="mb-2"
            intent="primary"
            fullWidth
            onClick={handleSaveSettings}
          >
            Apply settings
          </Button>
        </Accordion>
      </div>

      {/* <Button callback={handleAutoDetect} text="Auto detect" /> */}
    </div>
  );
};

export default Toolbar;
