import Link from "next/link";
import Image from 'next/image'
import githubIcon from "../static/icons/github-mark.png"

const Navbar = () => {
  return (
    <nav className="z-[100] h-[80px] pr-10 pl-10 bg-gray-200 flex justify-between items-center border-b-4 border-gray-800 ">
      <h1 className="text-bold text-6xl font-bold  text-gray-900   hover:cursor-pointer"> #PARKER </h1>


    <div className="flex gap-40">
    <Link
        className="text-bold text-4xl font-bold hover:underline"
        href={"/vision"}
      >
        {" "}
        Vision{" "}
      </Link>

      <Link className="text-bold text-4xl font-bold hover:underline" href={"/docs"}>
        {" "}
        Docs {" "}
      </Link>
    </div>



      <div className="flex justify-between gap-8 justify-center items-center ">
      <Image className="hover:cursor-pointer" width={35} src={githubIcon} height={35}/>
        <div className="bg-gray-900 pt-2 pb-2 pl-5 pr-5 group transition duration-300  "> 
          <span className="text-white text-2xl font-bold hover:underline transition-all leading-2 text-center w-2 hover:cursor-pointer "> SIGN UP</span>
        </div>

  
      </div>

    </nav>
  );
};

export default Navbar;
