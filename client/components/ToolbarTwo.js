import { imageHeightState, processingState} from "../components/states";
import { useRecoilValue , useRecoilState} from "recoil";
const ToolbarTwo = ({
  processing,
  setProcessing,
  hasWebcam,
  webcamEnabled,
  setWebcamEnable
}) => {


  const imageHeight = useRecoilValue(imageHeightState);

  function handleProcessing() {
    console.log(processing);
    processing ? setProcessing(false) : setProcessing(true);
  }
  function handleEnable() {
    webcamEnabled ? setWebcamEnable(false) : setWebcamEnable(true);
  }



  return (
    <div className={`w-[250px] min-h-[${imageHeight}px] `}>
      <div
        onClick={(e) => {
          if (hasWebcam) {
            
            handleEnable();
          }
        }}
        className={`
                items-center
              justify-center   cursor-pointer
                border border-gray-900
            
            p-2  focus:outline-none active:bg-blue-700"
                ${
                  webcamEnabled
                    ? " bg-slate-100 text-gray-800 duration-500 transition-colors     hover:bg-gray-300 hover:transition-none "
                    : " text-white  bg-red-500 duration-1000  transition-colors "
                }
         
                 text-gray-900`}
      >
        <p className="text font-bold text-center  pt-2 pb-2  ">
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
 
          handleProcessing()
        }}
        className={`
        items-center
      justify-center   cursor-pointer
        border border-gray-900
        p-2
        ${
          processing
            ? "bg-slate-100 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"
            : " bg-red-500 text-white duration-1000 transition-colors "
        }
 
         text-gray-900`}
      >
        <p className="text font-bold text-center  pt-2 pb-2  ">
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
      </div>

      {!hasWebcam && (
        <div
          onClick={(e) => {
            handleWebcamRefresh();
          }}
          className={`
        items-center
      justify-center   cursor-pointer
        border border-gray-900
        p-2
        ${"bg-slate-100 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"}
 
         text-gray-900`}
        >
          <p className="text font-bold text-center  pt-2 pb-2  ">
            <span> Reload Webcam </span>
          </p>
        </div>
      )}
    </div>
  );
};

export default ToolbarTwo;
