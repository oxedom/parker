import { useRecoilState } from "recoil";
import {
  selectingColorState,
  detectionColorState,
  processingState,
  selectedColorState,
  roiTypeState,
  roiNameState,
} from "./states";

const Toolbar = ({setWebCamApprove, webcamApprove}) => {


  const [selectedColor, setSelectedColor] = useRecoilState(selectedColorState);
  const [detectionColor, setDetectionColor] =
    useRecoilState(detectionColorState);
  const [selectingColor, setSelectingColor] =
    useRecoilState(selectingColorState);
  const [roiName, setRoiName] = useRecoilState(roiNameState);
  const [roiType, setRoiType] = useRecoilState(roiTypeState);
  const [processing, setProcessing] = useRecoilState(processingState);

  const handleColorChange = (e, stateChanger) => {
    stateChanger(e.target.value);
  };

  function handleProcessing() {
    console.log(webcamApprove);
    if(webcamApprove) 
    {
      processing ? setProcessing(false) : setProcessing(true);
    }
    }

  function handleEnable() 
  {
    webcamApprove ? setWebCamApprove(false) : setWebCamApprove(true);
   } 
   

  return (
    <div className="  flex  rounded-l-lg">
      <div className="flex flex-col-reverse gap-5">
        <div
          onClick={(e) => {
    
            handleProcessing();
          }}
          className={`rounded
             relative inline-flex group items-center
              justify-center   m-1 px-5 cursor-pointer
                shadow-lg
              
                ${webcamApprove ? "bg-blue-500 " : " text-gray-900  bg-gray-50 grayscale cursor-not-allowed" }
                 text-white`}
        >
          <span className="text font-bold text-center  pt-2 pb-2">Process Video</span>
        </div>

        <div onClick={(e) => {handleEnable()}} className={`rounded
             relative inline-flex group items-center m-1
              justify-center  px-5    cursor-pointer
                shadow-lg 
                
                ${webcamApprove ? "bg-green-50 " : " bg-red-500  " }
         
                 text-gray-900`}>

          <span className="text font-bold text-center  pt-2 pb-2  "> ENABLE VIDEO </span>
        </div>


        <div></div>
      </div>
    </div>
  );
};

export default Toolbar;
