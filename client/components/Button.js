

const Button = ({ text, callback, colors }) => {


  let audio = new Audio("../static/click.wav")

  const handleSound = () => 
  {

    audio.play()
  }

  const btnClass = `items-center
    justify-center   cursor-pointer
    duration-300
      rounded-lg 
     
        ${colors.color}
      transition-colors ease-linear
        hover:brightness-125
        duration-200
        shadow-neo
        outline-[1.5px]
        m-4
        w-[150px]
        outline
        outline-black
        p-4
       text-gray-800`;

  const paraClass = `text font-bold text-center    ${colors.textColor}  pt-2 pb-2`;

  return (
    <div
      onClick={(e) => {
        callback();
        handleSound();
      }}
      className={btnClass}
    >
      <p className={`${paraClass}`}> {text}</p>
    </div>
  );
};

export default Button;
