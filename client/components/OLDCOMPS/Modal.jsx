import React from "react";

const Modal = ({ isOpen, closeModal, children }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed p-10 top-20 left-0 right-0 bottom-0 bg-gray-500 bg-opacity-75 z-10"
      onClick={closeModal}
    >
      {children}
    </div>
  );
};

export default Modal;
