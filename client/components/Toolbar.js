import {
  imageHeightState,
  detectionThresholdState,
  thresholdIouState,
  fpsState,
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
  showDetections,
  setShowDetections,
  loadedCoco,
  setWebcamEnable,
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
  const [detectionThreshold, setDetectonThreshold] = useRecoilState(
    detectionThresholdState
  );
  const [settingChange, setSettingsChange] = useState(false)
  const [iouThreshold, setIouThreshold] = useRecoilState(thresholdIouState);
  const [fps, setFps] = useRecoilState(fpsState);

  const [localDetectonThreshold, setLocalDetectonThreshold] =
    useState(undefined);
  const [localIouThreshold, setLocalIouThreshold] = useState(undefined);
  const [localFps, setLocalFps] = useState(undefined);

  useEffect(() => {
    setLocalFps(fps);
    setLocalIouThreshold(iouThreshold * 100);
    setLocalDetectonThreshold(detectionThreshold * 100);
  }, [isModalOpen]);

  const handleSaveSettings = () => {
    // closeModal
    if(!settingChange) { return;}
    let wasProcessing = processing;

    setDetectonThreshold(localDetectonThreshold / 100);
    setIouThreshold(localIouThreshold / 100);
    setFps(localFps);
    setTimeout(() => {
      setProcessing(false);
    }, 0);
    setTimeout(() => {
      setProcessing(true);
    }, 0);
    // setProcessing(true)
    closeModal();
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
              <Button
                text={` ${
                  showDetections ? "View Detections" : " Hide detections "
                }`}
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

              <div></div>
              <Button
                text={"Reload Webcam"}
                colors={{ color: "bg-red-500", hover: "bg-red-200" }}
                callback={handleWebcamRefresh}
              />
            </div>

            <div className="flex flex-col justify-center items-center  ">
              <Decrementor
                alt="detection score threshold"
                step={1}
                min={0}
                max={99}
                value={localDetectonThreshold}
                setter={(value) => {  setLocalDetectonThreshold(value); setSettingsChange(true) } }
                label="Detection Threshold"
              />

              <Decrementor
                alt="iou threshold"
                step={1}
                min={0}
                max={99}
                value={localIouThreshold}
                label="iou  Threshold"
                setter={(value) => { setLocalIouThreshold(value); setSettingsChange(true) } }
              />

              <Decrementor
                alt="fps"
                step={100}
                min={100}
                max={10000}
                value={localFps}
                label="FPS"
                setter={(value) => { setLocalFps(value); setSettingsChange(true) } }
              />

              <Button
                colors={{ color: `${settingChange ? "bg-blue-500" : "bg-blue-300 cursor-not-allowed"}`, textColor: "text-white" }}
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
      <Button
        colors={{ color: "bg-slate-200", hover: "bg-slate-100" }}
        text="Auto detect"
      />
    </div>
  );
};

export default Toolbar;
