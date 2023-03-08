import Link from "next/link";
import Image from "next/image";
import githubIcon from "../public/static/icons/github-mark.png";
import Navlink from "./Navlink.js";
import Router from "next/router";

const Navbar = () => {
  const handleVision = () => {
    console.log(Router.route);
    if (Router.route == "/vision") {
      window.location.reload();
    }
  };
  return (
    <div className="hidden lg:block w-full text-center box-border w-max-[1028px] ">
      <nav className="z-[100]  h-[100px] absolute   ">
        <div className="flex gap-2 justify-center items-center">
          <Link href={"/"}>
            <h1 className=" text-bold text-3xl     justify-center items-center text-center  font-bold  text-white    hover:cursor-pointer">
              <p className=" p-2 ml-5 uppercase flex justify-items-center items-center gap-5 ">
                {" "}
                Parker
              </p>
            </h1>
          </Link>

          <div className="flex justify-self-center  items-center px-5 gap-10 ">
            <div
              onClick={handleVision}
              className="border-2  px-3 hover:scale-105 duration-200 rounded border-orange-600"
            >
              <Navlink url={"/vision"} text={"Try now"} />
            </div>

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
