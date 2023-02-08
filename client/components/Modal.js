import React from 'react';

const Modal = ({ isOpen, closeModal, children }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 bg-gray-500 bg-opacity-75">
      <div className="m-auto w-1/3 h-64 bg-white p-8">
        {children}
        <button className="mt-4" onClick={closeModal}>
          Close Modal
        </button>
      </div>
    </div>
  );
};

export default Modal;
