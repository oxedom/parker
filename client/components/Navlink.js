import Link from "next/link";
import { useRouter } from "next/router";

const Navlink = ({ url, text }) => {
  const router = useRouter();
  let selectedPathname = () => {
    if (router.route == url) {
      return "bg-purple-500";
    } else {
      return "bg-orange-400";
    }
  };

  return (
    <div className="flex ">
      <Link
        className={`text-bold   
        rounded-xl   shadow-neo  
          
         hover:shadow-neo-hover 
         sm:w-[100px]
         h-[50px]
         outline-[1.5px]
         outline
         

         outline-black
         inline-block  shadow-outline
         text-white font-bold ${selectedPathname()}`}
        href={url}
      >
        <span className="text-center inline-block w-full"> {text} </span>
      </Link>
    </div>
  );
};

export default Navlink;
