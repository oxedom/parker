import Link from "next/link";
import { useRouter } from "next/router";

const Navlink = ({ url, text }) => {
  const router = useRouter();
  let selectedPathname = () => {
    if (router.route == url) {
      return "bg-purple-500";
    } else {
      ("text-gray-700");
    }
  };

  return (
    <div>
      <Link
        className={`text-bold text-2xl  
        outline-[1.5px]
        outline
        outline-black
         bg-orange-400 rounded-2xl p-3 shadow-neo  black
         hover:shadow-neo-hover 
         inline-block  shadow-outline
         text-white font-bold ${selectedPathname()}`}
        href={url}
      >
        <span

        >
          {" "}
          {text}{" "}
        </span>
      </Link>
    </div>
  );
};

export default Navlink;
