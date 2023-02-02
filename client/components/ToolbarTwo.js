import { imageHeightState, processingState } from "../components/states";
import { useRecoilValue, useRecoilState } from "recoil";
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

     text-gray-900`

  const paraClass = 'text font-bold text-center  pt-2 pb-2'

  function handleProcessing() {
    processing ? setProcessing(false) : setProcessing(true);
  }
  function handleWebcamEnable() {
    webcamEnabled ? setWebcamEnable(false) : setWebcamEnable(true);
  }

  function handleDetectionsEnable() {
    showDetections ? setShowDetections(false) : setShowDetections(true);
  }




  function handleWebcamRefresh() {
    setHasWebcam(false);
  }

  return (
    <div className={`w-[200px] min-h-[${imageHeight}px] `}>
      <div
        onClick={(e) => {
          if (hasWebcam) {
            handleWebcamEnable();
          }
        }}
        className={`
        ${btnClass}
                ${
                  webcamEnabled
                    ? "bg-green-400 text-gray-800 duration-500 transition-colors     hover:transition-none "
                    : " text-white  bg-red-500 duration-1000  transition-colors "
                }
         
                `}
      >
        <p className={paraClass}>
          {webcamEnabled ? (
            <>
              <span>Webcam Enabled</span>
            </>
          ) : (
            <>
              <span>Webcam Disabled</span>
            </>
          )}
        </p>
      </div>

      <div
        onClick={(e) => {
          handleProcessing();
        }}
        className={`
        ${
          processing
            ? "bg-green-400 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"
            : " bg-red-500 text-white duration-1000 transition-colors "
        }
 
         ${btnClass}`}
      >
        <p className={paraClass}>
          {processing ? (
            <>
              <span>Mointoring Enabled</span>
            </>
          ) : (
            <>
              <span>Mointoring Disabled</span>
            </>
          )}
        </p>

<label class="relative inline-flex items-center mb-5 cursor-pointer">
  <input type="checkbox" value="" class="sr-only peer"/>
  <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>

</label>

      </div>



      <div
        onClick={(e) => { if(processing) {handleDetectionsEnable()}}}
        className={`
        ${
          showDetections && processing
            ? "bg-green-400 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"
            : " bg-red-500 text-white duration-1000 transition-colors "
        }
 
         ${btnClass}`}
        >
        <p className={paraClass}>
          <span> Show detections</span>
        </p>
      </div>

      <div
        onClick={(e) => {}}
        className={ `${btnClass} bg-gray-100 `}
        >
        <p className={paraClass}>
          <span> Auto parking</span>
        </p>
      </div>

      <div
        onClick={(e) => {
          handleWebcamRefresh();
        }}
        className={ `${btnClass} bg-gray-100 `}
      >
        <p className={paraClass}>
          <span> Reload Webcam </span>
        </p>
      </div>
    </div>
  );
};

export default ToolbarTwo;
