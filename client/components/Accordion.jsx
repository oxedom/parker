import { useState } from "react";

function Accordion({children, title, imageHeight}) {
  const [isOpen, setIsOpen] = useState(true);

  function toggleAccordion() {
    setIsOpen(!isOpen);
  }

  return (
    <div className={`max-w-sm  max-h-[${imageHeight}px] mx-auto`}>
      <div
        className={`flex justify-around items-center   ${isOpen ? "border-b  border-gray-200" : ""}  cursor-pointer`}
        onClick={toggleAccordion}
      >
        <h3 className="text-xl text-center font-medium text-white">{title}</h3>
        <svg
          className={`w-6 h-6 transition-transform invert duration-300 transform  ${
            isOpen ? "rotate-180" : ""
          }`}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>
      {isOpen && (
        <div className=" ">
          <ul className="flex flex-col text-center gap-2">
              {children.map(setting => { return <li> {setting} </li>})}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Accordion;