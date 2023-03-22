const ToogleSwitch = ({ boolean, callback, text }) => {
  function handleToogle() {
    callback();
  }

  return (
    <div className="flex  flex-col items-center">
      <span className=" font-bold mr-3   drop-shadow-sm text-white ">{text}</span>
      <label
        onChange={(e) => {
          handleToogle();
        }}
        className="flex items-around relative w-max cursor-pointer select-none"
      >
        <input
          value={boolean}
          type="checkbox"
          checked={boolean}
          className="  checked:bg-green-500 appearance-none transition-colors cursor-pointer w-14 h-7   rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black focus:ring-blue-500 bg-red-500"
        />
        <span className="absolute  font-medium text-xs uppercase right-1  text-white">
          {" "}
        </span>
        <span className="absolute font-medium text-xs uppercase right-8 text-white">
          {" "}
        </span>
        <span
          className={`w-7 h-7 right-7 absolute rounded-full transform transition-transform bg-gray-200 ${
            boolean ? "translate-x-7" : ""
          } `}
        />
      </label>
    </div>
  );
};

export default ToogleSwitch;
