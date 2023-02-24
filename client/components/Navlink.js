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
    <div>
      <Link
        className={`text-bold text-2xl  
        rounded-2xl p-3 shadow-neo  
         hover:shadow-neo-hover 
         outline-[1.5px]
         outline
         outline-black
         inline-block  shadow-outline
         text-white font-bold ${selectedPathname()}`}
        href={url}
      >
        <span> {text} </span>
      </Link>
    </div>
  );
};

export default Navlink;
