import { debounce } from "lodash";

const Toolbar = ({
  fps,
  selectingBoxColor,
  selectedBoxColor,
  setRoiType,
  setRoiName,
  setFps,
  roiName,
  setSelectingBoxColor,
  setSelectedBoxColor,
}) => {
  const handleColorChange = (e, stateChanger) => {stateChanger(e.target.value);};

  return (
    <div className="border-8 p-1  w-80 flex  rounded-l-lg">
      <div className="flex flex-col ">
        <h2 className="bg-white text-3xl border-black "> Toolbar </h2>

        <div>
          <p> Selecting Color</p>
          <input
            value={selectingBoxColor}
            type="color"
            onChange={(e) => {
              handleColorChange(e, setSelectingBoxColor);
            }}
          />
        </div>

        <div>
          <p> Selected Color</p>
          <input
            value={selectedBoxColor}
            type="color"
            onChange={(e) => {
              handleColorChange(e, setSelectedBoxColor);
            }}
          />
        </div>

        <input
          onChange={(e) => {
            setRoiName(e.target.value);
          }}
          value={roiName}
          placeholder={"ROI name"}
          className="border-2 border-black"
        />

        <div className="flex">
          <span> FPS: </span>
          <input
            type="number"
            value={fps}
            onChange={(e) => {
              setFps(e.target.value);
            }}
            className="border-2 border-black"
            placeholder="FPS"
          />
        </div>

        <div className="rounded relative inline-flex group items-center justify-center px-3.5 py-2 m-1 cursor-pointer border-b-4 border-l-2 active:border-purple-600 active:shadow-none shadow-lg bg-gradient-to-tr from-purple-600 to-purple-500 border-purple-700 text-white">
          {" "}
          <span class="relative">Button Text</span>
        </div>
      </div>
    </div>
  );
};

export default Toolbar;
