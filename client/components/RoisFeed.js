import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageHeightState, evaluateTimeState } from "./states";
import deleteIcon from "../static/icons/delete_bin_black.png";
import loadingIcon from "../static/icons/loading.png";
import Button from "./Button";
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

  function printDate(time, evaluateTime) {
    //MS MILAsecoudns
    //S secounds
    //M Minutes

    let diff_ms = Math.floor(Date.now() - time);
    let diff_s = Math.floor(diff_ms / 1000);
    let diff_m = Math.floor(diff_s / 60);
    let diff_h = Math.floor(diff_ms / 60);
    if (diff_s < 60) {
      return `${diff_s} secounds`;
    }
    if (diff_s < 3600) {
      return `${diff_m} mintues`;
    }
    if (diff_s > 3600) {
      return `${diff_h} hours`;
    }
  }

  return (
    <div className={`w-[200px] bg-blue-300   min-h-[${imageHeight}px]`}>
      <h4
        className="text-xl text-center font-semibold  
        bg-blue-100
      p-4 m-4 text-gray-800 cursor-default rounded-lg border-slate-900 select-none shadow-neo 
       outline-1 outline outline-black
       "
      >
        {" "}
        Marked parking spaces
      </h4>
      <div className="flex flex-wrap gap-2 m-2">
        {selectedRegions.map((s) => (
          <div
            key={s.uid}
            onMouseOver={(e) => {
              handleSelect(s.uid);
            }}
            onMouseLeave={(e) => {
              handleUnselect(s.uid);
            }}
            value={s.uid}
            onClick={(e) => {
              handleRoiDelete(s.uid);
            }}
            className={`  h-10 w-10
            btn  font-semibold  hover:bg-red-600 transition-colors rounded shadow-neo-sm duration-300 ease-in-out border  
                drop-shadow
                border-gray-500
              ${s.evaluating ? "bg-gray-400 animate-pulse duration-1000" : ""}
              ${
                s.occupied ? "bg-red-500" : "bg-green-500"
              }   cursor-default  duration-100  border-slate-900 items-center justify-between`}
          >
            {s.hover ? (
              <img
                className="invert ease-in duration-200  opacity-0 hover:opacity-90 "
                src={deleteIcon.src}
              />
            ) : (
              ""
            )}
          </div>
        ))}
      </div>
              <button> Delete Sections </button>
              <button> Save Selections </button>
    </div>
  );
};

export default RoisFeed;
