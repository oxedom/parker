import Link from "next/link";
import { useRouter } from 'next/router';

const Navlink = ({ url, text}) => {

  const router = useRouter();
  let selectedPathname = () => 
  {
    console.log(router);
    if(router.route == url) 
    {
      return 'text-blue-500'
    }
    else 
    {
      'text-gray-700'
    }

  }

  return (
    <div>
      <Link
        className={`text-bold text-2xl group transition duration-300 font-bold ${selectedPathname()}`}
        href={url}
      >
        <span
          className="relative before:content-[''] before:absolute before:block before:w-full before:h-[2px] 
              before:bottom-0 before:left-0 before:bg-black
              drop-shadow 
              before:hover:scale-x-100 before:scale-x-0 before:origin-top-left
              before:transition before:ease-in-out before:duration-300 "
              
        >
          {" "}
          {text}{" "}
        </span>
      </Link>
    </div>
  );
};

export default Navlink;
