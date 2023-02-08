import { imageHeightState, processingState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import settingsIcon from "../static/icons/settings.png"
import ToogleButton from "./ToogleButton";
import Modal from "./Modal";
import { useState } from "react";
const ToolbarTwo = ({
  processing,
  setProcessing,
  hasWebcam,
  setHasWebcam,
  webcamEnabled,
  showDetections,
  setShowDetections,
  setWebcamEnable,
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };


  const btnClass = `items-center
  justify-center   cursor-pointer
    border border-gray-900
    p-2
    duration-100 transition-colors   hover:bg-gray-300 hover:transition-none

     text-gray-900`;

  const paraClass = "text font-bold text-center  pt-2 pb-2";

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
    if (processing) {
      showDetections ? setShowDetections(false) : setShowDetections(true);
    }
  }

  function handleWebcamRefresh() {
    setHasWebcam(false);
  }

  return (
    <div className={`w-[200px] min-h-[${imageHeight}px]  `}>
      <img
      onClick={openModal}
      className="cursor-pointer"
      src={settingsIcon.src}/>

      <Modal isOpen={isModalOpen} closeModal={closeModal}>
      
  

      <ToogleButton
        title={"Webcam enabled"}
        callback={handleWebcamEnable}
        state={webcamEnabled}
      />

      <ToogleButton
        title={"Mointor enabled"}
        callback={handleProcessing}
        state={processing}
      />

      <ToogleButton
        title={"Show detections"}
        callback={handleDetectionsEnable}
        state={showDetections}
      />



      <div
        onClick={(e) => {
          handleWebcamRefresh();
        }}
        className={`${btnClass} bg-gray-100 `}
      >
        <p className={paraClass}>
          <span> Reload Webcam </span>
        </p>
      </div>



      <p>
        Display settings
      </p>
      <p>
        FPS
      </p>
      </Modal>


      <div onClick={(e) => {}} className={`${btnClass} bg-gray-100 `}>
        <p className={paraClass}>
          <span> Auto detect</span>
        </p>
      </div>
    </div>
  );
};

export default ToolbarTwo;
