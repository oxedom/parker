const Button = ({text, callback, colors}) => {

    
    const btnClass = `items-center
    justify-center   cursor-pointer
      rounded-lg 
      
        ${colors.color}
      transition-colors duration-200 ease-linear
        hover:brightness-125
        shadow-neo
        outline-[1.5px]
        m-4
        outline
        outline-black
        p-4
       text-gray-800`;

    const paraClass = "text font-bold text-center  pt-2 pb-2";


    return ( <div onClick={(e) => {  callback() }} className={btnClass}>
         <p className={`${paraClass}`} > {text}</p>
    </div> );
}
 
export default Button;