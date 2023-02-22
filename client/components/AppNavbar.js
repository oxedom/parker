import { useRecoilValue } from "recoil";
import { selectedRoiState } from "./states";


const AppNavbar = ({
    openModal}) => {

    let btnClass = 
    "shadow-neooutline-[1.5px] border-l border-black text-bold p-4  duration-100   "
    const selectedRois = useRecoilValue(selectedRoiState);

    const savedRois = localStorage.getItem('savedRois');
    const hasSaved = () => { 
        if(savedRois === null) { return false}
        else { return true}
    }

    const handleSave = () => 
    {
        if(selectedRois.length > 0 ) 
        {
            console.log(selectedRois);
        }
    }

    return ( <nav className="bg-blue-300  border-black border mb-4 grid gap-2 grid-cols-2 "> 
    <div></div>
    <div className=" grid grid-cols-3 grid-flow-row
    ">
    <div></div>
    <button  onClick={handleSave}  className={`${btnClass}  ${selectedRois.length > 0 ? "bg-gray-200  " : "bg-gray-400 cursor-not-allowed text-slate-800"}` }> Save  </button>
    <button onClick={openModal} className={`${btnClass}  bg-gray-200   ` }> Settings </button>
    {!hasSaved ? <button  className={`${btnClass}  ${hasSaved() ? "bg-green-500 " : "bg-gray-400 cursor-not-allowed text-slate-800"}` }> Import </button> : <></> }

    </div>

    </nav> );
}
 
export default AppNavbar;