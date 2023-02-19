import { imageHeightState, thresholdScoreState, thresholdIouState, fpsState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import settingsIcon from "../static/icons/settings.png";
import ToogleButton from "./ToogleButton";
import Modal from "./Modal";
import { useState } from "react";
const Toolbar = ({
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
  const [threshold, setThreshold] = useRecoilState(thresholdScoreState)
  const [iouThreshold, setIouThreshold] = useRecoilState(thresholdIouState)
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [fps, setFps] = useRecoilState(fpsState)

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const btnClass = `items-center
  justify-center   cursor-pointer
    rounded-lg m-2
    transition-colors duration-200 ease-linear
         hover:bg-gray-300 hover:transition-none

     text-gray-800`;

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
        className="cursor-pointer invert transition duration-500 transform hover:rotate-90"
        src={settingsIcon.src}
      />

      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        <div
          className="flex flex-col justify-between m-auto w-1/3 h-2/3 bg-white p-8 z-30 "
          onClick={(e) => {
            e.stopPropagation();
          }}
        >
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
          <label> threshold for detections score</label>
          <input onChange={(e) => {
     
            setThreshold(e.target.value)}} alt="detection score threshold" step={0.05} min={0} max={0.99} type="number" value={threshold}/>  

            <label> threshold for iou</label>
            <input onChange={(e) => {
              
              setIouThreshold(e.target.value)}} alt="iou  threshold" step={0.05} min={0} max={0.99} type="number" value={iouThreshold}/>  

              <div>
              <input onChange={(e) => {
              
              setFps(e.target.value)}} alt="fps" step={100} min={0} max={10000} type="number" value={fps}/>  

              <span> {1/(fps / 1000)}</span>

              </div>


          <div



            onClick={(e) => {
              handleWebcamRefresh();
            }}
            className={`${btnClass} bg-gray-100 `}
          >
            <p className={`${paraClass} select-none`}>
              <span className=""> Reload Webcam </span>
            </p>
          </div>




          <p>Display settings</p>
          <p>FPS</p>
          <button className="mt-4 bg-red-500 " onClick={(e) => closeModal()}>
            Close settings
          </button>




        </div>



      </Modal>

      <div onClick={(e) => {}} className={`${btnClass} bg-gray-100 `}>
        <p className={paraClass}>
          <span> Auto detect</span>
        </p>
      </div>
    </div>
  );
};

export default Toolbar;
