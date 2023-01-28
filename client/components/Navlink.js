import Link from "next/link";

const Navlink = ({ url, text }) => {
  return (
    <div>
      <Link
        className="text-bold text-4xl group transition duration-300 font-bold "
        href={url}
      >
        <span
          className="relative before:content-[''] before:absolute before:block before:w-full before:h-[2px] 
              before:bottom-0 before:left-0 before:bg-black
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
