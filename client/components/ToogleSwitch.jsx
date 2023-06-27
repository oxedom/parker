const ToogleSwitch = ({ boolean, callback, text }) => {
  function handleToogle() {
    callback();
  }

  return (
    <div className="flex flex-col">
      <span className="pb-1 mr-3 font-bold text-left text-white drop-shadow-sm">
        {text}
      </span>
      <label
        onChange={(e) => {
          handleToogle();
        }}
        className="relative flex items-center cursor-pointer select-none w-max"
      >
        <input
          value={boolean}
          type="checkbox"
          checked={boolean}
          className="transition-colors bg-red-500 rounded-full appearance-none cursor-pointer checked:bg-green-500 w-14 h-7 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black focus:ring-blue-500"
        />
        <span
          className={`w-5 h-5 right-8 absolute rounded-full transform transition-transform bg-gray-200 ${
            boolean ? "translate-x-7" : ""
          } `}
        />
      </label>
    </div>
  );
};

export default ToogleSwitch;
