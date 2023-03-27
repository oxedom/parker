import Image from "next/image";
import DefaultLayout from "../layouts/DefaultLayout";
import tensorflowIcon from "../public/static/tesnorflow.png";
import nextjsIcon from "../public/static/nextjs.png";
import webRTC from "../public/static/webRTC.png";
import github from "../public/static/icons/github-mark.png"
import Head from "next/head";
import yoloLogo from "../public/static/yolo.png"
import tailwindLogo from "../public/static/Tailwind.png"
import recoilLogo from "../public/static/Recoil.png"

const About = () => {
  return (
    <DefaultLayout>
      <Head>
      <title> About Parker</title>
      </Head>
      <div className="flex-1 flex mx-auto flex-col p-10 max-w-[800px] gap-5 mt-5items-stretch  ">
        <main className="">
          <div className="text-lg  ">
            <h2 className="text-5xl text-center flex  justify-center items-center gap-10 font-bold  rounded-lg  mb-10    text-orange-600">
              {" "}
              About parker
              <div>

                <a href="https://github.com/oxedom/parker" className="  ">               <Image width={50} src={github}></Image>    </a>
              </div>
            </h2>
            
            <section className="p-5  rounded-md mt-2 flex gap-2 flex-col text-xl  ">
         

              <p className="b-t border-black">
                {" "}
                <span className="text-2xl text-gray-700 font-bold">
                  Parker{" "}
                </span>{" "}
                is a free online smart parking tool that to enable users to privately (P2P )
                mointor parking spaces using their phones or webcams as cameras.  
              </p>

              <p className="text-4xl my-5   text-gray-700 ">Built with</p>

              <ul className="grid grid-cols-2 gap-2">
                <li className="flex gap-2">
                  <Image
                    alt="tensorflow"
                    width={30}
                    src={tensorflowIcon}
                  ></Image>
                  <a
                    className="text-blue-500"
                    href="https://www.tensorflow.org/js"
                  >
                    {" "}
                    TensorFlow.js
                  </a>
                </li>

                <li className="flex gap-2">
                  <Image width={30} alt="nextjs" src={nextjsIcon}></Image>
                  <span>NextJS </span>
                </li>

                <li className="flex gap-2">
                  <Image width={30} alt="webrtc" src={webRTC}></Image>
                  <span>WebRTC (PeerJS) </span>
                </li>

                
        

         
                <li className="flex gap-2 text-center items-center">
                  {" "}
                  <Image width={60} src={yoloLogo}></Image>
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                     YOLO7-tfjs  (Thanks Hugo){" "}
                  </a>{" "}
                  
                </li>
    
                <li className="flex gap-2 text-center items-center">
                  {" "}
                  <Image width={30} src={tailwindLogo}></Image>
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                   tailwindcss
                  </a>{" "}
                  
                </li>

                <li className="flex gap-2 text-center items-center">
                  {" "}
                  <Image width={60} src={recoilLogo}></Image>
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                    Recoil 
                  </a>{" "}
                  
                </li>
         
              </ul>
            </section>
          </div>
        </main>
      </div>
    </DefaultLayout>
  );
};

export default About;
