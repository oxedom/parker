const StreamSettings = ({WebRTCMode, peerId}) => {
  const linkStyle = "bg-blue-500 ";

  return (
    <div className="flex p-10 fixed z-9999 justify-center items-center">
      {WebRTCMode ? <>
        {/* <p> Stream ID: {peerId} </p> */}
      <a href={`http://localhost:3000/input?remoteID=${peerId}`} className={linkStyle}> Input Link </a>
      {/* <p className={linkStyle}> View Live Link</p> */}
      
      </> : null}

    </div>
  );
};

export default StreamSettings;
