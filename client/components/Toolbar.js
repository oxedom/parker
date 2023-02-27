import {
  imageHeightState,
  detectionThresholdState,
  thresholdIouState,
  autoDetectState,
  selectedRoiState,
  fpsState,
  showDetectionsState,
} from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import Modal from "./Modal";
import { useEffect, useState } from "react";
import Button from "./Button";
import Decrementor from "./Decrementor";

const Toolbar = ({
  processing,
  setProcessing,
  openModal,
  closeModal,
  isModalOpen,
  setHasWebcam,
  webcamEnabled,
  loadedCoco,
  setWebcamEnable,
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
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
  const [localIouThreshold, setLocalIouThreshold] = useState(undefined);
  const [localFps, setLocalFps] = useState(undefined);
  const selectedRois = useRecoilValue(selectedRoiState);

  const totalOccupied = (roiArr) => {
    let OccupiedCount = 0;
    let availableCount = 0;
    roiArr.forEach((roi) => {
      if (roi.occupied === true) {
        OccupiedCount++;
      }
      if (roi.occupied === false) {
        availableCount++;
      }
    });
    return { OccupiedCount, availableCount };
  };

  let counts = totalOccupied(selectedRois);
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
  }, [isModalOpen]);

  const handleAutoDetect = () => {
    if (autoDetect) {
      setAutoDetect(false);
    } else {
      setAutoDetect(true);
    }
  };

  const handleSaveSettings = () => {
    // closeModal
    if (!settingChange) {
      return;
    }
    let wasProcessing = processing;

    setDetectonThreshold(localDetectionThreshold / 100);
    setIouThreshold(localIouThreshold / 100);
    setFps(localFps);

    setProcessing(false);

    // setProcessing(true)
    closeModal();
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
  function handleWebcamEnable() {
    webcamEnabled ? setWebcamEnable(false) : setWebcamEnable(true);
  }

  function handleDetectionsEnable() {
    if (processing && loadedCoco) {
      showDetections ? setShowDetections(false) : setShowDetections(true);
    }
  }

  function handleWebcamRefresh() {
    if (loadedCoco) {
      setHasWebcam(false);
    }
  }

  return (
    <div className={`w-[200px]   min-h-[${imageHeight}px]  bg-blue-300  `}>
      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        <div
          className="flex flex-col justify-between m-auto w-5/12 xl:w-6/12 bg-emerald-50 p-8 z-30 rounded-lg shadow-neo "
          onClick={(e) => {
            e.stopPropagation();
          }}
        >
          <h1 className="text-center text-4xl text-white font-bold p-5 bg-indigo-500 border-b-1 rounded-full">
            {" "}
            Settings
          </h1>
          <div className="grid grid-cols-2">
            <div className="grid grid-rows-3 grid-cols-2  justify-center">
              <Button
                text={`${webcamEnabled ? "Webcam Enabled" : "Webcam Disabled"}`}
                callback={handleWebcamEnable}
                colors={{
                  color: `${webcamEnabled ? "bg-green-500" : "bg-red-500"}`,
                }}
              />
              <Button
                text={`${
                  processing ? "Processing Enabled" : "Processing Disabled "
                }`}
                callback={handleProcessing}
                colors={{
                  color: `${processing ? "bg-green-500" : "bg-red-500"}`,
                }}
              />

              <div></div>
              <Button
                text={"Reload Webcam"}
                colors={{ color: "bg-red-500", hover: "bg-red-200" }}
                callback={handleWebcamRefresh}
              />
            </div>

            <div className="flex flex-col justify-center items-center  ">
              <Decrementor
                information={detectInfo}
                alt="detection score threshold"
                step={1}
                min={10}
                max={100}
                value={localDetectionThreshold}
                setter={(value) => {
                  setLocalDetectionThreshold(value);
                  setSettingsChange(true);
                }}
                label="Detection Threshold"
              />

              <Decrementor
                information={iouInfo}
                alt="iou threshold"
                step={1}
                min={10}
                max={100}
                value={localIouThreshold}
                label="iou  Threshold"
                setter={(value) => {
                  setLocalIouThreshold(value);
                  setSettingsChange(true);
                }}
              />

              <Decrementor
                information={fpsInfo}
                alt="fps"
                step={0.01}
                min={0.1}
                max={60}
                value={localFps}
                label={`Render rate`}
                setter={(value) => {
                  setLocalFps(value);
                  setSettingsChange(true);
                }}
              />

              <Button
                colors={{
                  color: `${
                    settingChange
                      ? "bg-blue-500"
                      : "bg-blue-300 hover:cursor-not-allowed"
                  }`,
                  textColor: "text-white",
                }}
                callback={handleSaveSettings}
                text={"Save settings"}
              >
                {" "}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 justify-items-center border-t-2  border-black ">
            <Button
              colors={{ color: "bg-slate-50" }}
              callback={closeModal}
              text={"Exit "}
            >
              {" "}
            </Button>
          </div>

  

        </div>
      </Modal>
      <Button text={"Settings"} callback={openModal}  colors={{ color: "bg-slate-200", hover: "bg-slate-100" }}/>


      <Button
        colors={{ color: "bg-slate-200", hover: "bg-slate-100" }}
        callback={handleAutoDetect}
        text="Auto detect"
      />

      <Button
        text={` ${showDetections ? "View Detections" : " Hide detections "}`}
        callback={handleDetectionsEnable}
        colors={{
          color: `

        ${
          showDetections
            ? `${processing ? "bg-green-500" : "bg-gray-500"}`
            : `${processing ? "bg-red-500" : "bg-gray-500"}`
        }

        `,
        }}
      />

      <h6>{` Total spaces: ${selectedRois.length}`}</h6>
      <h6>{` Total free spaces: ${counts.availableCount}`}</h6>
      <h6>{` Total occupied spaces: ${counts.OccupiedCount}`}</h6>
    </div>
  );
};

export default Toolbar;
