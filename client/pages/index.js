import DefaultLayout from "../layouts/DefaultLayout";
import Image
 from "next/image";
import eye from '../static/eye.png'
import svg from '../public/bg1.svg'
import githubIcon from "../static/icons/github-mark.png";
import Head from "next/head";

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

          <header className="grid grid-cols-2 items-center text-white   place-items-center max-h-[1000px]  m-auto py-10 px-20 ">



          <h2 className=" text-6xl font-bold ">      
            <div className="grid grid-temp-col-2">
            <span>
              Turn your webcam into <br/>
          a parking sensor. </span>   
         <Image className="inline-block" width={10} height={50} src={eye}></ Image>   
            </div>

            <p className="text-2xl font-normal"> Using your webcam can turn your view into <br/> a smart computer vision 
            parking mointor  </p>

            <button className="text-2xl duration-300  rounded-2xl  bg-orange-600 p-4 hover:scale-105"> Try parker </button>
           </h2>

          <div>


           <video
              className="rounded-3xl object-contain"
                    // autoPlay
                     muted={true}
                     loop={true}
                     src="./test.mp4"
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
