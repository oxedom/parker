import { useState } from "react";

function Accordion({ children, title, imageHeight }) {
  const [isOpen, setIsOpen] = useState(true);

  function toggleAccordion() {
    setIsOpen(!isOpen);
  }

  return (
    <div className="w-full px-3 mx-auto">
      <div
        className={`flex duration-200 justify-between py-2 mb-2 items-center ${
          isOpen ? "border-b  border-gray-200" : ""
        }  cursor-pointer`}
        onClick={toggleAccordion}
      >
        <h3 className="text-2xl font-medium text-white">{title}</h3>

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
        <div className="duration-75 animate-fade ">
          <ul className="flex flex-col gap-2 text-center">
            {children.map((setting, index) => {
              return <li key={index}> {setting} </li>;
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Accordion;
