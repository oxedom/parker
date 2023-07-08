import React from "react";

const Modal = ({ isOpen, closeModal, children }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="absolute z-10 left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2  "
      onClick={closeModal}
    >
      <div className="relative">
      <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 "> 
      {children}

      </div>
      </div>

     


    </div>
  );
};

export default Modal;
