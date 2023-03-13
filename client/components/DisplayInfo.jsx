import { totalOccupied } from "../libs/utillity";
import { useRecoilValue } from "recoil";
import { selectedRoiState } from "./states";

const DisplayInfo = () => {
  const selectedRois = useRecoilValue(selectedRoiState);
  const counts = totalOccupied(selectedRois);
  return (
    <div className="flex gap-5 justify-center items-center place-content-center self-center">
      <div className="flex items-center flex-col gap-2">
        <p className="border-b-4 border-green-700 rounded">Available</p>
        <span>{counts.availableCount}</span>
      </div>
      <div className="flex items-center flex-col gap-2">
        <p className="border-b-4 border-red-700 rounded ">Occupied</p>
        <span className=" ">{counts.OccupiedCount}</span>
      </div>
    </div>
  );
};

export default DisplayInfo;
