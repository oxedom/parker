const ToolbarTwo = ({
  setWebCamApproved,
  webcamApproved,
  setProcessing,
  setAttempt,
  processing,
  hasWebcam,
}) => {
  function handleProcessing() {
    processing ? setProcessing(false) : setProcessing(true);
  }
  function handleEnable() {
    webcamApproved ? setWebCamApproved(false) : setWebCamApproved(true);
  }

  function handleWebcamRefresh() 
  {
    setAttempt((prev => { return prev+1}))
  }

  return (
    <div className="  w-[250px]">
      <div
        onClick={(e) => {
          if (hasWebcam) {
            handleWebcamRefresh();
            handleEnable();
          }
        }}
        className={`
                items-center
              justify-center   cursor-pointer
                border border-gray-900
            
            p-2  focus:outline-none active:bg-blue-700"
                ${
                  webcamApproved
                    ? " bg-slate-100 text-gray-800 duration-500 transition-colors     hover:bg-gray-300 hover:transition-none "
                    : " text-white  bg-red-500 duration-1000  transition-colors "
                }
         
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
          if (webcamApproved) {
            handleWebcamRefresh();
            handleProcessing();
            if(webcamApproved) {
              handleProcessing(false)
             }
          }
        }}
        className={`
        items-center
      justify-center   cursor-pointer
        border border-gray-900
        p-2
        ${
          processing
            ? "bg-slate-100 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"
            : " bg-red-500 text-white duration-1000 transition-colors "
        }
 
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
        
        {!hasWebcam && (
      <div
        onClick={(e) => {
      
            handleWebcamRefresh();
          
        }}
        className={`
        items-center
      justify-center   cursor-pointer
        border border-gray-900
        p-2
        ${
          
            "bg-slate-100 duration-500 transition-colors   hover:bg-gray-300 hover:transition-none"
          
        }
 
         text-gray-900`}
      >
        <p className="text font-bold text-center  pt-2 pb-2  ">
          <span> Reload Webcam </span>
        </p>
      </div> ) }
    </div>
  );
};

export default ToolbarTwo;
