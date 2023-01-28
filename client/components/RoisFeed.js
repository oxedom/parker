import { selectedRoiState } from "./states";
import { useRecoilValue, useRecoilState } from "recoil";

const RoisFeed = ({}) => {
  const [selectedRegions, setSelectedRois] = useRecoilState(selectedRoiState);

  function handleRoiDelete(uid) {
    let action = {
      event: "deleteRoi",
      payload: uid,
    };
    setSelectedRois(action);
  }

  return (
    <div className="w-[250px]">
      <h4 className="text-3xl text-center font-semibold text-gray-800 border-gray-800 border-b-2 border-l-2   bg-amber-50 p-5 ">
        {" "}
        ROI FEED
      </h4>
      <div className="overflow-y-scroll ">
        {selectedRegions.map((s) => (
          <div key={s.uid} className="flex gap-5">
            <h1>{s.name}</h1>
            <div
              value={s.uid}
              onClick={(e) => {
                handleRoiDelete(s.uid);
              }}
              className="btn hover:bg-red-600 bg-red-500 p-4 text-white text-xl font-medium tracking-wider"
            >
              Delete{" "}
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};

export default RoisFeed;
