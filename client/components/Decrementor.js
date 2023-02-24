import { useState } from "react";

const Decrementor = ({ min, max, step, setter, value, label }) => {
  const [id_up, setId_up] = useState(undefined);
  const [id_down, setId_down] = useState(undefined);

  const btnClass = "p-5 w-[50px] h-[50px] rounded shadow";
  function handleIncrement(e) {
    let afterIncrement = Math.round(parseInt(e.target.value) + step);
    if (afterIncrement < max) {
      setter(afterIncrement);
    }
  }

  function handleDecrement(e) {
    let afterDecrement = Math.round(parseInt(e.target.value) - step);
    if (afterDecrement > min) {
      setter(afterDecrement);
    }
  }
  // onClick={(e) => {handleDecrement(e)}}
  // onClick={(e) => {handleIncrement(e)}}
  return (
    <div className="flex flex-col pt-1 items-center">
      <label
        className="
        border m-1
        "
      >
        {" "}
        {label}{" "}
      </label>
      <div className="grid grid-cols-3 items-center justify-center justify-items-center">
        <button
          value={value}
          label="decrease"
          onMouseUp={() => {
            clearInterval(id_down);
          }}
          onMouseDown={(e) => {
            setId_down(
              setInterval(() => {
                handleDecrement(e);
              }, 50)
            );
          }}
          className={`${btnClass} bg-gray-300`}
        >
          -
        </button>

        <span className=" text-lg border-gray-200 text-center "> {value}</span>
        <button
          value={value}
          label="increase"
          onMouseUp={() => {
            clearInterval(id_up);
          }}
          onMouseDown={(e) => {
            setId_up(
              setInterval(() => {
                handleIncrement(e);
              }, 50)
            );
          }}
          className={`${btnClass} bg-gray-400  `}
        >
          +
        </button>
      </div>
    </div>
  );
};

export default Decrementor;
