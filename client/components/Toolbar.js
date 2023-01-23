import { useRecoilState } from "recoil";
import {
  selectingColorState,
  detectionColorState,
  processingState,
  selectedColorState,
  roiTypeState,
  roiNameState,
} from "./states";

const Toolbar = () => {
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
    processing ? setProcessing(false) : setProcessing(true);
  }

  return (
    <div className="  flex  rounded-l-lg">
      <div className="flex flex-col gap-5">

      <div
            onClick={(e) => {
              handleProcessing();
            }}
            className="rounded
             relative inline-flex group items-center
              justify-center  py-2  cursor-pointer
                shadow-lg bg-blue-500
             
                 text-white"
          >
            {" "}
            <span class="static">Process Video</span>
          </div>



        <div className="flex flex-col gap-5 border-b-4 pb-2">
          <input
            onChange={(e) => {
              setRoiName(e.target.value);
            }}
            value={roiName}
            placeholder={"ROI name"}
            className="h-14 rounded-md"
          />

          <select
            onChange={(e) => {
              setRoiType(e.target.value);
            }}
            value={roiType}
            className="h-10 w-32"
          >
            <option> Any </option>
            <option> Person </option>
            <option> Car </option>
            <option> Cat </option>
          </select>
        </div>

        <div>



        </div>
      </div>
    </div>
  );
};

export default Toolbar;
