import Link from "next/link";
import { useRouter } from "next/router";

const Navlink = ({ url, text }) => {
  const router = useRouter();
  let selectedPathname = () => {
    const string = "";

    if (router.route == url && router.route != "vision") {
      return "text-white";
    } else {
      return "text-gray-200 hover:text-white";
    }
  };

  return (
    <div className="flex hover:scale-105 duration-200">
      <Link
        className={`text-bold   
          

        text-2xl  

         text-white font-bold ${selectedPathname()}`}
        href={url}
      >
        <span className="text-center inline-block w-full"> {text} </span>
      </Link>
    </div>
  );
};

export default Navlink;
