import { useRef } from "react";

const ToogleButton = ({ title, callback, state }) => {

  const inputRef = useRef(null)

  return (
    <div className=" gap-2">

      <label onClick={(e) => inputRef.current.click()} className=""> {title} </label>
      <div>
      <label name={title} className="relative inline-flex items-center  cursor-pointer">
        <input
          name={title}
          ref={inputRef}
          alt={title}
          type="checkbox"
          defaultChecked={state}
          onChange={(e) => {
            callback();
          }}
          class={`sr-only peer ${state ? "" : "checked"} `}
        />
        
        <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
      </label>
      </div>
    </div>
  );
};

export default ToogleButton;
