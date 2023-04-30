import { useState } from "react";

const Instruction = ({ title, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleOpen = () => setIsOpen(!isOpen);

  return (
    <div className="border border-gray-200 p-4 rounded-md mb-4">
      <button
        className="w-full text-left font-medium  mb-2 focus:outline-none"
        onClick={toggleOpen}
      >
        <span className="text-black text-2xl">{title}</span>
      </button>
      <div
        className={`overflow-hidden transition-height duration-300 ${
          isOpen ? "h-auto" : "h-0"
        }`}
      >
        <div>{children}</div>
      </div>
    </div>
  );
};

export default Instruction;
