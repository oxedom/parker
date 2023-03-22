import DisplayInfo from "./DisplayInfo";



const VisionHeader = ({setWebRTCMode, setAllowWebcam, demo, handleDisableDemo, WebRTCMode, webcamEnable}) => {

        const btnStyle = "p-3 bg-orange-500 rounded-lg mx-2 "

    return ( <nav className="flex justify-around">
                 {demo ? (
                <div onClick={handleDisableDemo}>
                  <h5 className="font-bold p-4  rounded-lg text-gray-200 justify-self-start hover:text-white bg-orange-600   text-2xl  duration-300  ">
                    {" "}
                    Exit{" "}
                  </h5>
                  <span className=""> </span>
                </div>
              ) : (
           
                <>
         
                <div className="grid grid-cols-3 gap-2 items-center content-center mx-2   justify-center   duration-600  transition ease-in   rounded-lg ">
                  <button
                           className={btnStyle}
                    onClick={(e) => {
                        setWebRTCMode(false);
                      setAllowWebcam(true);
                    }}
                  >
                    {" "}
                    <span className=""> Webcam </span>
                  </button>

                  <button
                    className={btnStyle}
                    onClick={(e) => {
                        setAllowWebcam(false);
                      setWebRTCMode(true);
                    }}
                  >
                    {" "}
                    <span className="">RTC  </span>
                  </button>

                
              <DisplayInfo></DisplayInfo> 
  
         
                </div>
       
             
                </>)}
    </nav>  );
}
 
export default VisionHeader;