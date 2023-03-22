const StreamSettings = ({WebRTCMode, peerId, WebRTCLoaded}) => {
  const btnStyle = "bg-gray-200 shadow-sm text-lg  text-slate-700  duration-1000 font-semibold hover:bg-gray-300 hover:text-slate-800  border border-black p-5 rounded  ";

  const handleCopy = () => 
  {
    navigator.clipboard.writeText(`https://www.sam-brink.com/input?remoteID=${peerId}`);
  }

  return (    
    
    <div className="  my-8  ">
      {WebRTCMode  ? <>
        {/* <p> Stream ID: {peerId} </p> */}
        <div className="fixed flex-col my-3 justify-center items-center   z-999">
        {/* <button alt="streaming Link"  className={btnStyle}  onClick={handleCopy}> Copy Link </button> */}
        <button alt="streaming Link"  className={btnStyle}  onClick={handleCopy}> Copy Link </button>

        </div>

      
      </> : null}

    </div>
  );
};

export default StreamSettings;
