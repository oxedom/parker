const Button = ({ text, callback, colors }) => {
  const btnClass = `items-center
    justify-center   cursor-pointer

      rounded-lg 
     
     
      transition-colors ease-linear
        hover:brightness-125
   
    
 
        outline-[1.5px]
        m-4
        w-max-w-[150px]
        outline
        outline-black
        text-white
     `;

  const paraClass = `text font-bold text-center       duration-100   hover:scale-110   pt-2 pb-2`;

  return (
    <div
      onClick={(e) => {
        callback();
        // handleSound();
      }}
      className={btnClass}
    >
      <p className={`${paraClass}`}> {text}</p>
    </div>
  );
};

export default Button;
