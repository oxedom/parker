import { imageHeightState, processingState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import ToogleButton from "./ToogleButton";
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

      <div onClick={(e) => {}} className={`${btnClass} bg-gray-100 `}>
        <p className={paraClass}>
          <span> Auto parking</span>
        </p>
      </div>

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
    </div>
  );
};

export default ToolbarTwo;
