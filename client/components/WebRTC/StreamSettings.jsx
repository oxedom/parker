const StreamSettings = () => {
  const linkStyle = "bg-blue-500 p-1";

  return (
    <div className="flex  justify-center items-center">
      <p> Stream ID: 112321132</p>
      <p className={linkStyle}> Input Link</p>
      <p className={linkStyle}> View Live Link</p>
    </div>
  );
};

export default StreamSettings;
