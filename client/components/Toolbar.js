import { useRecoilState } from "recoil";
import {selectingColorState, selectedColorColorState} from './states'
const Toolbar = ({fps, classes,selectingBoxColor, selectedBoxColor, setRoiType, setRoiName, setFps, roiName,}) => {


  const [selectedColor,setSelectedColor] = useRecoilState(selectedColorColorState);
  const [selectingColor, setSelectingColor] = useRecoilState(selectingColorState);

  const handleColorChange = (e, stateChanger) => {
    stateChanger(e.target.value);
  };

  return (
    <div className="border-8 p-1  w-80 flex  rounded-l-lg">
      <div className="flex flex-col gap-5">
        <h2 className="bg-white text-3xl border-b-4 border-black ">
          {" "}
          Toolbar{" "}
        </h2>

        <div>
          <div className="border-b-4 border-black ">
            <input
              value={selectingColor}
              type="color"
              className="h-20 w-20"
              onChange={(e) => {
                handleColorChange(e, setSelectingColor);
              }}
            />

            <input
              className="h-20 w-20"
              value={selectedColor}
              type="color"
              onChange={(e) => {
                handleColorChange(e, setSelectedColor);
              }}
            />
          </div>
        </div>

        <div className="flex flex-col gap-5 border-b-4 pb-2 border-black">
          <input
            onChange={(e) => {
              setRoiName(e.target.value);
            }}
            value={roiName}
            placeholder={"ROI name"}
            className="border-2 border-black h-5 rounded-md"
          />

          <select className="h-10 w-32">
            <option> Person </option>
            <option> Car </option>
            <option> Cat </option>
          </select>
        </div>

        <div>
          <div className="flex">
            <label> FPMS: </label>
            <input
              type="number"
              value={fps}
              onChange={(e) => {
                setFps(e.target.value);
              }}
              className="border-2 border-black h-8 rounded-md"
              placeholder="FPS"
            />
          </div>

          <div className="rounded relative inline-flex group items-center justify-center px-3.5 py-2 m-1 cursor-pointer border-b-4 border-l-2 active:border-blue-600 active:shadow-none shadow-lg bg-gradient-to-tr from-blue-500 to-cyan-400 border-sky-700 text-white">
            {" "}
            <span class="static">Process Video</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Toolbar;
