import Link from "next/link";
import Image from "next/image";
import githubIcon from "../static/icons/github-mark.png";
import Navlink from "./Navlink.js";

const Navbar = () => {
  return (
    <div className="w-full text-center box-border w-max-[1028px] ">
      <nav className="z-[100] h-[100px] relative  bg-yellow-100  items-center  grid grid-cols-2   ">
        <Link href={"/"} >
          <h1 className="text-bold text-5xl h-[50px] flex  justify-center items-center text-center  font-bold  text-black    hover:cursor-pointer">

        <p className=" p-2 uppercase flex justify-items-center items-center gap-5 "> #Parker 
          <span className="border rounded-full shadow-neo-sm bg-purple-200 p-1 text-xl text-center"> Beta</span> 
          </p>
    
          </h1>
        </Link>

        <div className="flex gap-4 justify-self-center  h-full items-center px-5 ">
          <div className="  flex justify-center items-center h-full">
            <Navlink url={"/vision"} text={"Try now!"} />
          </div>
          <div className="  flex justify-center items-center h-full ">
            <Navlink
              className="  

"
              url={"/about"}
              text={"About"}
            />
          </div>

          <div className="  flex justify-center items-center h-full ">
            <Navlink
              className="  

"
              url={"/docs"}
              text={"Getting Started"}
            />
          </div>
        </div>

        {/* <div className="flex justify-between gap-8 justify-self-end   h-full items-center ">
          <Image
            alt="githubIcon"
            className="hover:cursor-pointer "
            width={35}
            src={githubIcon}
            height={35}
          />
        </div> */}


      </nav>
    </div>
  );
};

export default Navbar;
