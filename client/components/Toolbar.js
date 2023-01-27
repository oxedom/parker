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
              justify-center  py-2 p-10  cursor-pointer
                shadow-lg
                bg-blue-400
                ${webcamApprove ? "bg-blue-500 " : "grayscale cursor-not-allowed" }
                 text-white`}
        >
          <span class="static">Process Video</span>
        </div>

        <div onClick={(e) => {handleEnable()}} className={`rounded
             relative inline-flex group items-center
              justify-center  py-2 p-10  cursor-pointer
                shadow-lg 
                
                ${webcamApprove ? "bg-green-500 " : "animate-pulse bg-blue-500 " }
         
                 text-white`}>

          <span children="static"> Enable Video </span>
        </div>


        <div></div>
      </div>
    </div>
  );
};

export default Toolbar;
