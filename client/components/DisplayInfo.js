import { totalOccupied } from "../libs/utillity";
import { useRecoilValue } from "recoil";
import { selectedRoiState } from "../components/states";

  
const DisplayInfo = () => {

      const selectedRois = useRecoilValue(selectedRoiState);
  const counts = totalOccupied(selectedRois)
    return ( 
    <div className="grid grid-cols-3 gap-2 justify-center items-center place-content-center self-center"> 
      <h6>{`  Available : ${counts.availableCount}`}</h6>
      <h6>{`  Occupied : ${counts.OccupiedCount}`}</h6>
                  
                
                   </div>
   );
}
 
export default DisplayInfo;