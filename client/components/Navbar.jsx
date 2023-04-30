import Link from "next/link";
import Navlink from "./Navlink.jsx";
import Router from "next/router";

const Navbar = () => {
  const handleVision = () => {
    if (Router.route == "/vision") {
      window.location.reload();
    }
  };
  return (
    <div className="my-5 sm:my-0 block w-full text-center w-max-[1028px]  h-[50px] sm:h-16   ">
      <nav className="z-[100]  absolute   ">
        <div className="flex gap-2 justify-center items-center">
          <Link href={"/"}>
            <h1 className=" text-bold text-3xl      justify-center items-center text-center  font-bold  text-white    hover:cursor-pointer">
              <p className=" p-2 ml-5 uppercase flex justify-items-center items-center gap-5 ">
                {" "}
                Parkerr
              </p>
            </h1>
          </Link>

          <div className="flex justify-self-center  items-center px-5 gap-5">
            <div
              onClick={handleVision}
              className="border-2 hidden sm:block  px-2 hover:scale-105 duration-200 rounded border-orange-600"
            >
              <Navlink url={"/vision"} text={"Try now"} />
            </div>

            <Navlink
              className="  

"
              url={"/about"}
              text={"About"}
            />
          </div>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
