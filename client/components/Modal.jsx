import React from "react";
import Button from "./Button";
const Modal = ({ isOpen, closeModal, children }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed top-1/2  -translate-y-1/2 flex items-center  z-[99999] justify-center bg-black bg-opacity-50">
      <div className="bg-filler p-3  rounded-xl ">
        {children}

        
        <Button  intent="destructive"  className="mt-2 "         onClick={closeModal}> <span className="text-lg "> Close Modal </span> </Button>
  
      </div>
    </div>
  );
};

export default Modal;
