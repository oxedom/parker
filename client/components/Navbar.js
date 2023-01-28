import Link from "next/link";
import Image from "next/image";
import githubIcon from "../static/icons/github-mark.png";
import Navlink from "./Navlink.js";

const Navbar = () => {
  return (
    <nav className="z-[100] h-[80px]  pl-10 pr-10 bg-gray-200 flex justify-between items-center outline outline-1 outline-stone-900  ">
      <h1 className="text-bold text-6xl font-bold  text-gray-900   hover:cursor-pointer">
        {" "}
        #PARKER{" "}
      </h1>

      <div className="flex gap-40 border-x-2  border-black h-full items-center px-5 ">
        <Navlink url={"/vision"} text={"VISON"} />
        <Navlink url={"/docs"} text={"DOCS"} />
      </div>

      <div className="flex justify-between gap-8   h-full items-center ">
        <div className="bg-gray-900 flex justify-center flex-col pl-5 pr-5 h-full  group transition duration-300  ">
          <p
            className="text-white text-2xl font-bold relative cursor-pointer before:content-[''] before:absolute before:block before:w-full before:h-[2px] 
              before:bottom-0 before:left-0 before:bg-white
              before:hover:scale-x-100 before:scale-x-0 before:origin-top-left
              before:transition before:ease-in-out before:duration-300"
          >
            {" "}
            SIGN UP
          </p>
        </div>
        <Image
          className="hover:cursor-pointer"
          width={35}
          src={githubIcon}
          height={35}
        />
      </div>
    </nav>
  );
};

export default Navbar;
