import { imageHeightState, thresholdScoreState, thresholdIouState, fpsState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import settingsIcon from "../static/icons/settings.png";
import ToogleButton from "./ToogleButton";
import Modal from "./Modal";
import { useState } from "react";
import Button from "./Button";
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
    <div className={`w-[200px] min-h-[${imageHeight}px]  bg-blue-300  `}>


      <Button text="Settings" callback={openModal} colors={{color: "bg-slate-200", hover: "bg-slate-100"} } />
      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        <div
          className="flex flex-col justify-between m-auto w-1/3 h-2/3 bg-white p-8 z-30 rounded-lg shadow-neo "
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
              
              setFps(e.target.value)}} alt="fps" step={1000} min={0} max={10000} type="number"/>  

              <span> {1/(fps / 1000)} FPS </span>

              </div>

                <Button  text={"Reload Webcam"} colors={{color: "bg-red-500", hover: "bg-red-200"} } callback={handleWebcamRefresh}/> 







          <p>Display settings</p>
          <p>FPS</p>
          <div className="flex gap-2 justify-center">
          <Button colors={{color: "bg-blue-500", hover: "bg-blue-200"} }  callback={closeModal} text={'Save settings'}> </Button>
          <Button colors={{color: "bg-red-500", hover: "bg-red-200"} }   callback={closeModal} text={'Exit settings'}> </Button>

          </div>

    
       




        </div>



      </Modal>


            <Button  colors={{color: "bg-slate-200", hover: "bg-slate-100"}} text="Auto detect"/>



    </div>
  );
};

export default Toolbar;
