import Link from "next/link";
import Image from "next/image";
import githubIcon from "../static/icons/github-mark.png";
import Navlink from "./Navlink.js";

const Navbar = () => {
  return (
    <div className="hidden lg:block w-full text-center box-border w-max-[1028px] ">
      <nav className="z-[100]  h-[70px] relative  bg-yellow-200   ">
        <div className="flex justify-center items-center">
        <Link href={"/"}>

          <h1 className=" text-bold text-3xl     justify-center items-center text-center  font-bold  text-black    hover:cursor-pointer">
            <p className=" p-2 uppercase flex justify-items-center items-center gap-5 ">
              {" "}
              #Parker
 
            </p>
          </h1>
        </Link>

        <div className="flex gap-4 justify-self-center  items-center px-5 ">
          
            <Navlink url={"/vision"} text={"Try now!"} />
  


        
            <Navlink
              className="  

"
              url={"/about"}
              text={"About"}
            />
    

         
            <Navlink
              className="  

"
              url={"/docs"}
              text={"Docs"}
            />
     
        </div>
        </div>



      </nav>
    </div>
  );
};

export default Navbar;
