const Loader = ({ progress }) => {
  return (
    <div className="bg-pink-500 p-10 h-full w-full">
      <p> {progress * 100}% </p>
    </div>
  );
};

export default Loader;
