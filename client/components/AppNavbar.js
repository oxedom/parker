import { useEffect, useState } from "react";
import { useRecoilValue } from "recoil";
import { selectedRoiState } from "./states";

const AppNavbar = ({ openModal }) => {
  let btnClass =
    "shadow-neooutline-[1.5px] border-l border-black text-bold p-4  duration-100   ";

  const [hasImports, setHasImports] = useState(false);
  const selectedRois = useRecoilValue(selectedRoiState);

  useEffect(() => {
    if (localStorage.getItem("savedRois") != null) {
      setHasImports(true);
    }
  }, []);

  const handleImport = () => {
    const savedRois = JSON.parse(localStorage.getItem("savedRois"));
    console.log("banna");
    console.log(savedRois);
  };

  const handleSave = () => {
    if (selectedRois.length > 0) {
      let selectedRoisClone = structuredClone(selectedRois);

      for (let index = 0; index < selectedRoisClone.length; index++) {
        selectedRoisClone[index]["time"] = 0;
        selectedRoisClone[index]["savedDate"] = Date.now();
        selectedRoisClone[index]["evaluating"] = true;
        selectedRoisClone[index]["firstSeen"] = null;
        selectedRoisClone[index]["lastSeen"] = null;
        selectedRoisClone[index]["occupied"] = false;
        selectedRoisClone[index]["hover"] = false;
      }
      prompt("");
      localStorage.setItem("savedRois", JSON.stringify(selectedRoisClone));
      setHasImports(true);
    }
  };

  return (
    <nav className="bg-blue-300  border-black border mb-4 grid gap-2 grid-cols-2 ">
      <div className="grid grid-cols-3">
        <button className="bg-gray-200 p-4"> File </button>
      </div>

      <div
        className=" grid grid-cols-3 grid-flow-row
    "
      >
        <button
          onClick={handleSave}
          className={`${btnClass}  ${
            selectedRois.length > 0
              ? "bg-gray-200  "
              : "bg-gray-400 cursor-not-allowed text-slate-800"
          }`}
        >
          {" "}
          Save{" "}
        </button>
        <button onClick={openModal} className={`${btnClass}  bg-gray-200   `}>
          {" "}
          Settings{" "}
        </button>
        <button
          onClick={handleImport}
          className={`${btnClass}  ${
            hasImports ? "bg-gray-200 " : " bg-gray-400 cursor-not-allowed"
          }`}
        >
          {" "}
          Import{" "}
        </button>
      </div>
    </nav>
  );
};

export default AppNavbar;
