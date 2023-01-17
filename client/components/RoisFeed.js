
import { selectedRoiState} from './states'
import { useRecoilValue, useRecoilState } from "recoil";

const RoisFeed = () => {



  const [selectedRegions, setSelectedRois  ] = useRecoilState(selectedRoiState)
  function handleRoiDelete(uid) 
  {
    let action = 
    {
      event: 'deleteRoi',
      payload: uid
    }
    setSelectedRois(action)

  }


  
  return (
    <div className="border-2 border-indigo-600 w-80  rounded-r-lg flex flex-col">
      <h4 className="text-3xl border-2 border-black"> Selected ROI's</h4>

      {selectedRegions.map((s) => (
        <div className="flex ">
          <h1 key={s.uid}> {s.name}</h1>
          <div value={s.uid} onClick={(e)=> {(handleRoiDelete(s.uid))}} className="btn bg-red-500 p-4">DELETE </div>
        </div>
      ))}
    </div>
  );
};

export default RoisFeed;
