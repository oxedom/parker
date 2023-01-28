import { useRecoilState } from "recoil";
import { processingState } from "./states";

const ToolbarTwo = ({ setWebCamApproved, webcamApproved }) => {
  const [processing, setProcessing] = useRecoilState(processingState);

  function handleProcessing() {
    console.log(webcamApproved);
    if (webcamApproved) {
      processing ? setProcessing(false) : setProcessing(true);
    }
  }

  function handleEnable() {
    webcamApproved ? setWebCamApproved(false) : setWebCamApproved(true);
  }

  return (
    <div className="bg-green-500">
      <div
        onClick={(e) => {
          handleEnable();
        }}
        className={`
                items-center
              justify-center   cursor-pointer
                border border-gray-900
                
                ${webcamApproved ? "bg-slate-100 " : " bg-red-600  "}
         
                 text-gray-900`}
      >
        <p className="text font-bold text-center  pt-2 pb-2  ">
          {" "}
          ENABLE VIDEO{" "}
        </p>
      </div>

      <div
        onClick={(e) => {
          handleProcessing();
        }}
        className={`
             relative inline-flex group items-center
              justify-center    cursor-pointer
              border border-gray-900

                ${
                  webcamApproved
                    ? "bg-slate-100 text-gray-900 "
                    : "   bg-gray-300 text-gray-200 grayscale cursor-not-allowed"
                }
                 text-white`}
      >
        <span className="text-gray-900 font-bold text-center  pt-2 pb-2 cursor-default">
          PROCESS VIDEO{" "}
        </span>
      </div>
    </div>
  );
};

export default ToolbarTwo;
