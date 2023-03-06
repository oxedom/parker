import DefaultLayout from "../layouts/DefaultLayout";
import Image
 from "next/image";
import eye from '../static/eye.png'
import svg from '../public/bg1.svg'
import githubIcon from "../static/icons/github-mark.png";
import Head from "next/head";
import Link from "next/link";

const Home = () => {

  
  return (
    <>
    <Head>
      <title> Parker: Empower your webcam</title>
    </Head>
    <DefaultLayout>
      {/* Free SVG Background by <a target="_blank" href="https://bgjar.com">BGJar</a> */}
      <div className="">
     

   
        <main className=" ">

          <header className="flex text-clip grid-rows-none  items-center text-white
             place-items-center 
               md:m-auto py-20 px-20 ">



          <h2 className=" text-6xl mt-10 font-bold ">      
            <div className="grid grid-temp-col-2">
            <span className="">
              Turning your webcam into
          a parking mointor. </span>   
         <Image className="inline-block" width={10} height={50} src={eye}></ Image>   
            </div>

            <p className="text-xl  sm:text-2xl font-normal"> 
            Using the latest computer vision object detection models<br></br>to turn  your parking lot into  a smart one.  </p>

            <Link href={'/vision'} >
            <button className="text-2xl duration-300  rounded-2xl  bg-orange-600 p-4 hover:scale-105"> Use parker </button>

            </Link>
           </h2>

          <div>


           <video
              className="hidden lg:block rounded-3xl object-contain"
                    autoPlay
                    width={640}
                    height={480}
                     muted={true}
                     loop={true}
                     src="./demo.mp4"
          > </video>


          </div>
     

          </header>


          <div className="flex justify-center p-20">

    
          {/* <iframe width="640" height="360" src="https://www.youtube.com/embed/pIiNHX3uWzE" title="A Practical Exercise for Photographers" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> */}

          </div>
        </main>


 
      </div>
    </DefaultLayout>
    </>
  );
};

export default Home;
