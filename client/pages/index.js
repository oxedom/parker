
import { useState } from "react";
import Modal from "../components/Modal";

function Home() {

  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };


  return (
    <div>
      <h1>Welcome to my website!</h1>
      <button onClick={openModal}>Open Modal</button>
      <Modal isOpen={isModalOpen} closeModal={closeModal}>
        <p>This is a modal!</p>
      </Modal>
    </div>
  );
};

export default Home;
