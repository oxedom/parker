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
  closeModal,
  isModalOpen,
  allowWebcam,
  setAllowWebcam,
  loadedCoco,
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
  const [vehicleOnly, setVehicleOnly] = useRecoilState(
    vehicleOnlyState
  );
  const [detectionThreshold, setDetectonThreshold] = useRecoilState(
    detectionThresholdState
  );
  const [showDetections, setShowDetections] =
    useRecoilState(showDetectionsState);
  const [settingChange, setSettingsChange] = useState(false);
  const [iouThreshold, setIouThreshold] = useRecoilState(thresholdIouState);
  const [fps, setFps] = useRecoilState(fpsState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const [localDetectionThreshold, setLocalDetectionThreshold] = useState(undefined);
  const [localVehicleOnly, setLocalVehicleOnly] = useState(undefined)
  const [localIouThreshold, setLocalIouThreshold] = useState(undefined);
  const [localFps, setLocalFps] = useState(undefined);




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
    setLocalVehicleOnly(vehicleOnly)
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


      setDetectonThreshold(localDetectionThreshold / 100);
      setIouThreshold(localIouThreshold / 100);
      setFps(localFps);
      setVehicleOnly(localVehicleOnly)
      setProcessing(false);
      setSettingsChange(false)





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

    vehicleOnly ?  setLocalVehicleOnly(false) : setLocalVehicleOnly(true)
  }

  return (
    <div
      className={`md:w-[200px]  flex justify-between flex-col min-h-[${imageHeight}px]  bg-filler  `}
    >
      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        <div
          className="flex flex-col justify-between m-auto w-5/12 xl:w-6/12 bg-blue-800 -50 p-8 z-30 rounded-lg  "
          onClick={(e) => {
            e.stopPropagation();
          }}
        >
          <h1 className="text-center text-4xl text-white font-bold p-5 bg-orange-600 border-b-1 rounded-full">
            {" "}
            Settings
          </h1>
          <div className="grid grid-cols-2">
            <div className="grid grid-rows-2 gap-10 m-4">
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

              <div></div>
            </div>

            <div className="flex justify-center flex-col items-center ">

              <div className="flex   flex-col text-white justify-center grow">
                <label> Render Rate</label>
                <div className="grid grid-cols-2  ">
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

                  <span> {Math.floor(localFps * 100) / 100} FPS </span>
                </div>
              </div>
            </div>


            
            <div></div>
            <Button
              colors={{
                color: `${
                  settingChange
                    ? "bg-blue-500 p-5"
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

          <div className="grid grid-cols-2 justify-items-center border-t-2  border-black ">
            <Button
              colors={{ color: ", " }}
              callback={closeModal}
              text={"Exit "}
            >
              {" "}
            </Button>
          </div>
        </div>
      </Modal>

      <div className="flex justify-center flex-col items-center gap-2  ">
              <div className="flex flex-col text-white justify-center items-center grow">
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

              <div className="flex   flex-col text-white justify-center items-center grow ">
                <label> IOU Threshold</label>
                <div className="grid grid-cols-2  ">
                <span> {localIouThreshold}% </span>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    className="mr-4"
                    label="iou Threshold"
                    value={localIouThreshold}
                    onChange={(e) => {
                      setLocalIouThreshold(e.target.value);
                      setSettingsChange(true);
                    }}
                  />

      
                </div>
              </div>


              <Button
              colors={{
                color: `${
                  settingChange
                    ? "bg-blue-500 p-5"
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








      <Button callback={handleAutoDetect} text="Auto detect" />

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
    </div>
  );
};

export default Toolbar;
