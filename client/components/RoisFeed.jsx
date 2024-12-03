import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";
import { imageHeightState, autoDetectState } from "./states";
import deleteIcon from "../public/static/icons/delete_bin_black.png";
import Image from "next/image";
import Accordion from "./Accordion";
import Button from "./Button";



const RoisFeed = ({ openModal }) => {
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);
  const [autoDetect, setAutoDetect] = useRecoilState(autoDetectState);
  const imageHeight = useRecoilValue(imageHeightState);






  function handleDeleteAll() {
    if (selectedRegions.length === 0) {
      return;
    }
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


  function handleMangeData() {
    openModal()
  }



  return (
    <>

      <div
        className={`w-[200px] bg-black/60 backdrop-blur-sm rounded-xl justify-between min-h-[${imageHeight}px]`}
      >
        <div>
          <Accordion title={"Controls"}>
            <div className="flex flex-col gap-2 my-2">
              <Button intent="destructive" onClick={handleDeleteAll}>
                Delete regions
              </Button>

              <Button onClick={handleAutoDetect}>Auto Detect (Beta)</Button>


              <Button
                className={`
             hover:cursor-pointer drop-shadow
            font-bold border-white border rounded  shadow-black
            bg-white cursor-pointer text-slate-800"
        `}
                onClick={handleMangeData}
              >
                Manage Data
              </Button>

            </div>
            <div></div>
          </Accordion>

          <h4 className="text-xl font-semibold text-center text-white border-b-2 border-orange-600 hover:cursor-default ">
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
            btn  font-semibold  hover:bg-yellow-500 transition-colors rounded   border  
                drop-shadow
                border-gray-500
              ${s.evaluating ? "bg-gray-400 animate-pulse duration-1000" : ""}
              ${s.occupied ? "bg-red-500" : "bg-green-500"
                  }   cursor-default  duration-100  border-slate-900 items-center justify-between`}
              >
                {s.hover ? (
                  <Image
                    width={50}
                    height={50}
                    alt="Delete"
                    className="opacity-0 invert hover:opacity-90 "
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
    </>
  );
};

export default RoisFeed;
