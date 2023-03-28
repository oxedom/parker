import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageHeightState, evaluateTimeState, autoDetectState } from "./states";
import deleteIcon from "../public/static/icons/delete_bin_black.png";
import Image from "next/image";
import Accordion from "./Accordion";

const RoisFeed = ({}) => {
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const imageHeight = useRecoilValue(imageHeightState);
  const evaluateTime = useRecoilValue(evaluateTimeState);

  function handleDeleteAll() {
    if(selectedRegions.length === 0) { return;}
    let answer = confirm(
      "Are you sure you want to delete all selected regions?"
    );
    if (answer) {
      let action = {
        event: "deleteAllRois",
      };
      setSelectedRois(action);
    }
  }



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

  const handleAutoDetect = () => {
    if (!autoDetect) {
      setAutoDetect(true);
    }
  };

  const handleImport = () => {

    let action = {
      event: "importSelected",
      payload: null
    };
    setSelectedRois(action);
  }

  const handleSave = () => 
  {

    let selections = JSON.stringify(selectedRegions)
    localStorage.setItem('selections', selections)
    
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
      className={`w-[200px]  bg-orangeFadeSides rounded-xl    justify-between  min-h-[${imageHeight}px]`}
    >
      <div>
        <Accordion title={"Controls"}>
          <div className="flex flex-col my-2 gap-5">
            <button
              className={`${
                selectedRegions.length > 0
                  ? "bg-white text-slate-800  "
                  : "bg-gray-300   cursor-default text-gray-700 "
              }  font-bold p-2 rounded mx-2  `}
              onClick={handleDeleteAll}
            >
              {" "}
              Delete regions{" "}
            </button>

            <button
              className={`
            
            
             hover:cursor-pointer
            font-bold border-white border rounded p-2  shadow-black
            cursor-default
            ${
              autoDetect
                ? "bg-gray-300 text-gray-700 hover:cursor-default"
                : "bg-white cursor-pointer text-slate-800"
            }
        `}
              onClick={handleAutoDetect}
            >
              {" "}
              Auto Detect{" "}
            </button>

            <div className="grid grid-cols-2 gap-2 mx-1 place-content-between  ">
            <button
              className={`
            
            
             hover:cursor-pointer
            font-bold border-white border rounded   shadow-black
            cursor-default
            ${
              autoDetect
                ? "bg-gray-300 text-gray-700 hover:cursor-default"
                : "bg-white cursor-pointer text-slate-800"
            }
        `}
              onClick={handleSave}
            >
              {" "}
              Save  {" "}
            </button>

            <button
              className={`
            
            
             hover:cursor-pointer
            font-bold border-white border rounded  shadow-black
            cursor-default
            ${
              autoDetect
                ? "bg-gray-300 text-gray-700 hover:cursor-default"
                : "bg-white cursor-pointer text-slate-800"
            }
        `}
              onClick={handleImport}
            >
              {" "}
              Import  {" "}
            </button>
            </div>


          </div>
          <div></div>
        </Accordion>

        <h4
          className="text-xl text-center font-semibold  
      text-white hover:cursor-default
      border-b-2 border-orange-600


       "
        >
          {" "}
          Marked regions
        </h4>

        <div className="flex flex-wrap   gap-2 m-2">
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
            btn  font-semibold  hover:bg-yellow-500 transition-colors rounded   border  
                drop-shadow
                border-gray-500
              ${s.evaluating ? "bg-gray-400 animate-pulse duration-1000" : ""}
              ${
                s.occupied ? "bg-red-500" : "bg-green-500"
              }   cursor-default  duration-100  border-slate-900 items-center justify-between`}
            >
              {s.hover ? (
                <Image
                  width={50}
                  height={50}
                  alt="Delete"
                  className="invert  opacity-0 hover:opacity-90 "
                  src={deleteIcon.src}
                />
              ) : (
                ""
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RoisFeed;
