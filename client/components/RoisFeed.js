import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageHeightState, evaluateTimeState } from "./states";
import deleteIcon from "../public/static/icons/delete_bin_black.png";
import Image from "next/image";

const RoisFeed = ({}) => {
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);
  const imageHeight = useRecoilValue(imageHeightState);
  const evaluateTime = useRecoilValue(evaluateTimeState);

  function handleDeleteAll() {
    let action = {
      event: "deleteAllRois",
    };
    setSelectedRois(action);
  }

  function handleSave() {}

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
    <div
      className={`w-[200px]  bg-filler flex flex-col justify-between  min-h-[${imageHeight}px]`}
    >
      <div>
        <h4
          className="text-xl text-center font-semibold  
      text-white
      border-b-2 border-orange-600


       "
        >
          {" "}
          Marked regions
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
            btn  font-semibold  hover:bg-red-600 transition-colors rounded  duration-100 ease-in-out border  
                drop-shadow
                border-gray-500
              ${s.evaluating ? "bg-gray-400 animate-pulse duration-1000" : ""}
              ${
                s.occupied ? "bg-red-500" : "bg-green-500"
              }   cursor-default  duration-100  border-slate-900 items-center justify-between`}
            >
              {s.hover ? (
                <Image
                  alt="Delete "
                  className="invert ease-in duration-200  opacity-0 hover:opacity-90 "
                  src={deleteIcon.src}
                />
              ) : (
                ""
              )}
            </div>
          ))}
        </div>
      </div>

      <button
        className={`${
          selectedRegions.length > 0
            ? "bg-gray-200"
            : "bg-gray-400 cursor-not-allowed"
        } border-t border-l p-1  border-black `}
        onClick={handleDeleteAll}
      >
        {" "}
        Clear{" "}
      </button>

      {/* <button onClick={handleSave}> Save Selections </button> */}
    </div>
  );
};

export default RoisFeed;
