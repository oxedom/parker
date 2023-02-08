import React from 'react';

const Modal = ({ isOpen, closeModal, children }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed p-10 top-10 left-0 right-0 bottom-0 bg-gray-500 bg-opacity-75 z-10"
        onClick={closeModal} 
    >
      <div  
      className=" flex flex-col justify-between m-auto w-1/3 h-2/3 bg-white p-8 z-30"
      onClick={e => {e.stopPropagation()}}
      >

        {children}
        <button className="mt-4 bg-red-500 " onClick={(e) =>closeModal}>
          Close settings
        </button>

      </div>
    </div>
  );
};

export default Modal;
