import { useState } from "react";
import { Tooltip } from "react-tooltip";
const Decrementor = ({ min, max, step, setter, value, label, information }) => {
  const [id_up, setId_up] = useState(null);
  const [id_down, setId_down] = useState(null);

  const btnClass = "p-5 w-[50px] h-[50px] rounded shadow";
  function handleIncrement(e) {
    let afterIncrement = (parseFloat(e.target.value) + step).toFixed(2);

    if (afterIncrement <= max) {
      setter(afterIncrement);
    }
  }

  function handleDecrement(e) {
    let afterDecrement = (parseFloat(e.target.value) - step).toFixed(2);
    if (afterDecrement >= min) {
      setter(afterDecrement);
    }
  }

  return (
    <div className="">
      <div className="flex flex-col pt-1 items-center">
        <label
          className="
        border m-1
        "
        >
          {" "}
          {label}{" "}
        </label>

        <div className="grid grid-cols-4 items-center justify-center justify-items-center">
          <button
            value={value}
            label="decrease"
            onMouseUp={() => {
              if (setId_down != null) {
                clearInterval(id_down);
                setId_down(null);
              }
            }}
            onMouseDown={(e) => {
              setId_down(
                setInterval(() => {
                  handleDecrement(e);
                }, 20)
              );
            }}
            className={`${btnClass} bg-gray-300`}
          >
            -
          </button>

          <span className=" text-lg border-gray-200 text-center ">
            {" "}
            {value}
          </span>
          <button
            value={value}
            label="increase"
            onMouseUp={() => {
              if (setId_up != null) {
                clearInterval(id_up);
                setId_up(null);
              }
            }}
            onMouseDown={(e) => {
              setId_up(
                setInterval(() => {
                  handleIncrement(e);
                }, 20)
              );
            }}
            className={`${btnClass} bg-gray-400  `}
          >
            +
          </button>

          <div>
            <Tooltip id={label} />
            <p
              data-tooltip-id={label}
              data-tooltip-content={information}
              className="
          bg-blue-500
          p-2
          rounded
        "
            >
              i
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Decrementor;
