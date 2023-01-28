

const ToolbarTwo = ({ setWebCamApproved, webcamApproved, setProcessing, processing}) => {

  function handleProcessing() {
      processing ? setProcessing(false) : setProcessing(true);

  }
  function handleEnable() {
    webcamApproved ? setWebCamApproved(false) : setWebCamApproved(true);
  }

  return (
    <div className="  w-[250px]">
      <div
        onClick={(e) => {
          handleEnable();
        }}
        className={`
                items-center
              justify-center   cursor-pointer
                border border-gray-900
            p-2  focus:outline-none active:bg-blue-700"
                ${webcamApproved ? " bg-slate-100 text-gray-800 duration-500 transition-colors " : " text-white animate-pulse bg-red-400 duration-1000  transition-colors "}
         
                 text-gray-900`}
      >

          <p className="text font-bold text-center  pt-2 pb-2  ">
          {webcamApproved ? (
      <>
        <span>Webcam Enabled</span>

      </>
    ) : (
      <>
        <span>Webcam Disabled</span>

      </>
    )}
  
        </p>








      </div>

      <div
        onClick={(e) => {
          handleProcessing();
        }}
        className={`
        items-center
      justify-center   cursor-pointer
        border border-gray-900
    
        ${processing ? "bg-slate-100 duration-500 transition-colors " : " bg-red-600 duration-1000 transition-colors "}
 
         text-gray-900`}
      >
  <p className="text font-bold text-center  pt-2 pb-2  ">
  {processing ? (
      <>
        <span>Mointoring Enabled</span>

      </>
    ) : (
      <>
        <span>Mointoring Disabled</span>

      </>
    )}
        </p>
      </div>
    </div>
  );
};

export default ToolbarTwo;
