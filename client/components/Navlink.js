import Link from "next/link";
import { useRouter } from "next/router";

const Navlink = ({ url, text }) => {
  const router = useRouter();
  let selectedPathname = () => {
    if (router.route == url) {
      return "text-white";
    } else {
      return "text-white";
    }
  };

  return (
    <div className="flex hover:scale-105 duration-200">
      <Link
        className={`text-bold   
          

        text-xl

         text-white font-bold ${selectedPathname()}`}
        href={url}
      >
        <span className="text-center inline-block w-full"> {text} </span>
      </Link>
    </div>
  );
};

export default Navlink;
