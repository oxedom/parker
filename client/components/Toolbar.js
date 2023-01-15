

const Toolbar = ({
  fps,
  classes,
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
      <div className="flex flex-col gap-5">
        <h2 className="bg-white text-3xl border-black "> Toolbar </h2>

        <div className="flex gap-10 ">

          <div className="flex flex-col items-center">
          <p> Selecting Color</p>
          <input
            value={selectingBoxColor}
            type="color"
            className="h-20 w-20"
            onChange={(e) => {
              handleColorChange(e, setSelectingBoxColor);
            }}
          />
          </div>
 
          <div className="flex flex-col items-center"> 
          <p> Selected Color</p>
          <input
            className="h-20 w-20"
            value={selectedBoxColor}
            type="color"
            onChange={(e) => {
              handleColorChange(e, setSelectedBoxColor);
            }}
          />
        </div>


   

        </div>

        <div>
            <select>
              <option> 1 </option>
              <option> 2 </option>
              <option> 2 </option>
            </select>

            </div>

        <input
          onChange={(e) => {
            setRoiName(e.target.value);
          }}
          value={roiName}
          placeholder={"ROI name"}
          className="border-2 border-black h-8 rounded-md"
        />

        {/* <select>
          {classes.map((c) => <option> 1 </option>)}
        </select> */}

        <div className="flex">
          <span> FPMS: </span>
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
  );
};

export default Toolbar;

