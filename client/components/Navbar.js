import Link from "next/link";


const Navbar = () => {
  return (
    <nav className="z-[100] h-20 bg-indigo-200 flex justify-around items-center">
      <h1 className="text-bold text-3xl font-medium  text-white "> Parker </h1>
      <Link className="text-bold text-3xl font-medium" href={"/dashboard/parking"}> Parking </Link>
      <Link className="text-bold text-3xl font-medium" href={"/dashboard/"}> Home </Link>
      <p className="text-bold text-3xl font-medium  text-white"> Docs </p>

    </nav>
  );
};

export default Navbar;
