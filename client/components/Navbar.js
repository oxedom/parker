import Link from "next/link";
import Image from "next/image";
import githubIcon from "../static/icons/github-mark.png";
import Navlink from "./Navlink.js";

const Navbar = () => {
  return (
    <div className="w-full text-center box-border w-max-[1028px] border-b-[1px] border-gray-900/10">
      <nav className="z-[100] h-[75px] relative   pl-10 pr-10 bg-yellow-100  items-center  flex justify-between  ">
        <Link href={"/"}>
          <h1 className="text-bold text-5xl font-bold  text-gray-900     hover:cursor-pointer">
            {" "}
            #PARKER{" "}
          </h1>
        </Link>

        <div className="flex gap-40  h-full items-center px-5 ">
          <div className=" px-5  flex justify-center items-center h-full border-black">
            <Navlink url={"/vision"} text={"Try now"} />
          </div>
          <div className="  flex justify-center items-center h-full ">
            <Navlink
              className="  

"
              url={"/docs"}
              text={"Showcase"}
            />
          </div>
        </div>

        <div className="flex justify-between gap-8   h-full items-center ">
          <Image
            alt="githubIcon"
            className="hover:cursor-pointer"
            width={35}
            src={githubIcon}
            height={35}
          />
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
