import { imageHeightState, detectionThresholdState, thresholdIouState, fpsState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import Modal from "./Modal";
import { useEffect, useState } from "react";
import Button from "./Button";
const Toolbar = ({
  processing,
  setProcessing,
  hasWebcam,
  setHasWebcam,
  webcamEnabled,
  showDetections,
  setShowDetections,
  loadedCoco,
  setWebcamEnable,
  
}) => {
  const imageHeight = useRecoilValue(imageHeightState);
  const [detectionThreshold, setDetectonThreshold] = useRecoilState(detectionThresholdState)
  const [iouThreshold, setIouThreshold] = useRecoilState(thresholdIouState)
  const [fps, setFps] = useRecoilState(fpsState)


  const [isModalOpen, setIsModalOpen] = useState(false);
  const  [localDetectonThreshold, setLocalDetectonThreshold] = useState(undefined)
  const  [localIouThreshold, setLocalIouThreshold] = useState(undefined)
  const  [localFps, setLocalFps] = useState(undefined)

  useEffect(() => {
    setLocalFps(fps)
    setLocalIouThreshold(iouThreshold)
    setLocalDetectonThreshold(detectionThreshold)
  }, [])

  const openModal = () => {
    setIsModalOpen(true);
  };

  const handleSaveSettings = () => 
  {
    setDetectonThreshold(localDetectonThreshold)
    setIouThreshold((localIouThreshold))
    setFps(localFps)
  }

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
    if (processing && loadedCoco) {
      showDetections ? setShowDetections(false) : setShowDetections(true);
    } 
  }

  function handleWebcamRefresh() {
    if(loadedCoco) 
    {
      setHasWebcam(false);
    }

  }

  return (
    <div className={`w-[200px] min-h-[${imageHeight}px]  bg-blue-300  `}>


      <Button text="Settings" callback={openModal} colors={{color: "bg-slate-200", hover: "bg-slate-100"} } />
      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        
        <div
          className="flex flex-col justify-between m-auto w-1/4 h-3/3 bg-white p-8 z-30 rounded-lg shadow-neo "
          onClick={(e) => {
            e.stopPropagation();
          }}
        >

      <Button text={`${webcamEnabled ? "Disable Camera": "Enable Camera "}`} callback={handleWebcamEnable} colors={{color: `${webcamEnabled ? "bg-purple-500": "bg-orange-500"}`} }/>

      <Button text={`${processing ? "Stop processing": "Process "}`} callback={handleProcessing} colors={{color: `${processing ? "bg-purple-500": "bg-orange-500"}`} }/>

      <Button text={`${showDetections ? "Hide detections": "Show detections "}`} callback={handleDetectionsEnable} colors={{color: `${showDetections ? "bg-purple-500": "bg-orange-500"}`} }/>    
          {/* <ToogleButton
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
          /> */}

          <input onChange={(e) => {
     
           }} alt="detection score threshold" step={0.05} min={0} max={0.99} type="number" value={localDetectonThreshold}/>  

            <label> threshold for iou</label>
            <input onChange={(e) => {
              
            }} alt="iou  threshold" step={0.05} min={0} max={0.99} type="number" value={localIouThreshold}/>  

              <div>
              <input onChange={(e) => {
              
              setFps(e.target.value)}} alt="fps" step={1000} min={0} max={10000} type="number"/>  

              <span> {1/(localFps / 1000)} FPS </span>

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
