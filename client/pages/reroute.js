import { useRouter } from "next/router";

const reroute = () => {
  const router = useRouter();
  const handleReroute = (val) => {
    const currentDomain = new URL(window.location.href);

    if (val === "input") {
      currentDomain.pathname = "input";
      currentDomain.searchParams.set("remoteID", router.query.remoteID);

      router.push(currentDomain.href);
    }
    if (val === "view") {
      currentDomain.searchParams.set("remoteID", router.query.remoteID);

      currentDomain.pathname = "input";
      router.push(currentDomain.href);
    }
  };

  return (
    <div className="h-screen gap-2 pt-10 flex flex-col   min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-filler w-full grow items-center">
      <div className="flex flex-col  m-5 justify-center font-bold   gap-2">
        <header>
          <h2 className="text-center text-white  text-6xl  "> Menu </h2>
        </header>
        <button
          onClick={(e) => {
            handleReroute("input");
          }}
          className="bg-purple-500 rounded-md text-white text-3xl p-5"
        >
          {" "}
          Stream Video
        </button>
        <button
          onClick={(e) => {
            handleReroute("view");
          }}
          className="bg-orange-500 rounded-md text-white text-3xl p-5"
        >
          {" "}
          View Parking Situation
        </button>
      </div>
    </div>
  );
};

export default reroute;
