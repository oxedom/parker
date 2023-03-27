
import Head from "next/head";
import Link from "next/link";
import DashboardLayout from "../layouts/DashboardLayout";
const Home = () => {
  return (
    <>
      <Head>
        <title> Parker: Empower your parking </title>
        <link rel="apple-touch-icon" sizes="180x180" href="favicon/apple-touch-icon.png"/>
<link rel="icon" type="image/png" sizes="32x32" href="favicon/favicon-32x32.png"/>
<link rel="icon" type="image/png" sizes="16x16" href="favicon/favicon-16x16.png"/>
<link rel="manifest" href="/site.webmanifest"/>
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5"/>
<meta name="msapplication-TileColor" content="#da532c"/>

      </Head>
      <DashboardLayout>
        {/* Free SVG Background by <a target="_blank" href="https://bgjar.com">BGJar</a> */}
        <div className="">
          <main className=" ">
            <header
              className="flex text-clip grid-rows-none  items-center text-white
      
               md:p-0 md:py-20 px-10 md:px-20 "
            >
              <h2 className=" text-5xl md:mt-10 font-bold ">
                <div className="md:grid md:grid-temp-col-2">
                  <span className="animate-fade">
                    Turning any camera into a parking monitor .{" "}
                  </span>
                </div>
                
                <p className="text-xl  md:text-2xl font-normal">
                  Using computer vision and client side processing <br />
                  to turn your parking situation into a smart one.{" "}
                </p>

                <Link href={"/vision"}>
                  <button className="hidden sm:block my-5 text-2xl duration-300  rounded-lg  bg-orange-600 p-4 hover:scale-105">
                    {" "}
                    Use parker{" "}
                  </button>
                </Link>

                <Link href={"/about"}>
                  <button className="block sm:hidden my-5 text-2xl duration-300  rounded-lg  bg-orange-600 p-4 hover:scale-105">
                    {" "}
                    About parker{" "}
                  </button>
                </Link>
              </h2>

              <div>
                <video
                  className="hidden lg:block sm:mx-2 rounded-3xl object-contain animate-fade"
                  autoPlay
                  width={640}
                  height={480}
                  muted={true}
                  loop={true}
                  src="./demo.mp4"
                >
                  {" "}
                </video>
              </div>
            </header>
          </main>
        </div>
      </DashboardLayout>
    </>
  );
};

export default Home;
