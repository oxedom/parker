import Link from "next/link";

const Navbar = () => {
  return (
    <nav className="z-[100] h-20 bg-indigo-200 flex justify-around items-center">
      <h1 className="text-bold text-3xl font-medium  text-white "> Parker </h1>

      <Link className="text-bold text-3xl font-medium" href={"/"}>
        {" "}
        Home{" "}
      </Link>


      <Link
        className="text-bold text-3xl font-medium"
        href={"/vision"}
      >
        {" "}
        Vision{" "}
      </Link>

      <Link className="text-bold text-3xl font-medium" href={"/docs"}>
        {" "}
        Docs {" "}
      </Link>

    </nav>
  );
};

export default Navbar;
