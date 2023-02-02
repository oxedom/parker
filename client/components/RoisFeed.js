import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageHeightState, evaluateTimeState } from "./states";

const RoisFeed = ({}) => {
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);
  const imageHeight = useRecoilValue(imageHeightState);
  const evaluateTime = useRecoilValue(evaluateTimeState);

  function handleSelect(uid) {
    let action = {
      event: "selectRoi",
      payload: uid,
    };
    setSelectedRois(action);
  }

  function handleUnselect(uid) {
    let action = {
      event: "unSelectRoi",
      payload: uid,
    };
    setSelectedRois(action);
  }

  function handleRoiDelete(uid) {
    let action = {
      event: "deleteRoi",
      payload: uid,
    };
    setSelectedRois(action);
  }

  return (
    <div className={`w-[200px]   min-h-[${imageHeight}px]`}>
      <h4 className="text-3xl text-center font-semibold   bg-slate-100 p-2 text-gray-800 cursor-default border-b-2 border-r-2 border-slate-900">
        {" "}
        MARKED SPOTS
      </h4>
      <div className="">
        {selectedRegions.map((s) => (
          <div
            key={s.uid}
            onMouseOver={(e) => {
              handleSelect(s.uid);
            }}
            onMouseLeave={(e) => {
              handleUnselect(s.uid);
            }}
            className={
       
              `flex  text-xl w-full 
              ${Date.now() - s.time > evaluateTime ? "" : "bg-gray-300" } 
              
              ${(s.occupied && Date.now() - s.time > evaluateTime)  ? "bg-red-500" : "bg-green-500"}  cursor-default hover:bg-blue-400  border-b-2  border-slate-900 items-center justify-between`}
          >
            <p>
            {Date.now() - s.time > evaluateTime ? "" : "Evaluating" } 
            </p>
            <div
              value={s.uid}
              onClick={(e) => {
                handleRoiDelete(s.uid);
              }}
              className="btn hover:bg-gray-500 bg-slate-300 font-semibold  h-7 w-10 text-gray-900  text-xl border-l-2  border-r-2 border-slate-900   "
            >
              {" "}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RoisFeed;
