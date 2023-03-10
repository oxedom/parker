import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/router";

const NotFound = () => {
  const router = useRouter();

  useEffect(() => {
    setTimeout(() => {
      router.push("/");
    }, 0);
  }, []);
  return (
    
    <div className=" h-screen flex flex-col  items-center justify-around w-max-w-[1000px] min-h-screen w-max-w-[100px]  bg-fixed bg-no-repeat bg-cover  bg-hero"> 
    <div>
    <h3 className="text-white text-9xl text-center"> </h3>
        {/* <h4 className="text-white text-4xl"> This is not the parking space you are for...</h4> */}
    </div>
        
    </div>
  );
};

export default NotFound;
