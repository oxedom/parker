import { useRecoilState } from "recoil";
import {
  selectingColorState,
  detectionColorState,
  processingState,
  selectedColorState,
  roiTypeState,
  roiNameState,
} from "./states";

const Toolbar = ({ setWebCamApprove, webcamApprove }) => {
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
    if (webcamApprove) {
      processing ? setProcessing(false) : setProcessing(true);
    }
  }

  function handleEnable() {
    webcamApprove ? setWebCamApprove(false) : setWebCamApprove(true);
  }

  return (
    <div className="  flex  ">
      <div className="flex flex-col-reverse "></div>
    </div>
  );
};

export default Toolbar;
