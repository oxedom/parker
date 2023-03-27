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
import Accoridan from '../components/Accordion'
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

            <section className="flex flex-col gap-5">
              <h3 className="font-bold border-b border-black text-center text-3xl text-gray-800"> Useful information  </h3>
                
      

            <div className="mt-1 ">
            <h4 className="text-2xl font-bold ">
            Mobile Phone Camera Instructions
            </h4>
            <p>
            After entering the vision page and pressing the remote stream button, you can scan QR code with your mobile device and after the page loads press on the call button to allow acesss to your phones camera, two imporant notes that the phone should be in landscape mode and change your settings that your phone screen doesn't turn off after a mintue. 

            </p>

            </div>

            <div className="flex gap-2 flex-col">
            <h4 className="text-2xl font-bold   ">
            Connect CCTV/IP Cameras with OBS 
            </h4>
            <p>
            <strong> If your IP/CCTV cameras not are  not directly connected to your computer </strong>Stream their video footage using
            <a  className="text-blue-500 " href="https://obsproject.com/"> OBS </a>
           Window Capture feature and create a virtual webcam on your PC. 
            </p>
            <strong>
             If your IP/CCTV camera is on your local network   </strong> Set your OBS video capture setings to a local ip address.
            <a  className="text-blue-500" href="https://www.youtube.com/watch?v=0z9Te51rh-4"> Tutorial video</a>
            <p>

            </p>

            </div>

            <div className="mt-1 ">
            <h4 className="text-2xl font-bold ">
            Webcam Instructions
            </h4>
            <p>
            Open the vision page, press the webcam button, allow webcam access, and point it wherever you desire.
            </p>

            </div>

            <div className="mt-1 ">
            <h4 className="text-2xl font-bold ">
            What else can parker detect?
            </h4>
            <p>
              <a href="https://github.com/oxedom/parker/blob/main/client/libs/labels.json" className="text-blue-500"> Here is a list </a>
         
            </p>

            </div>
    
            </section>
          </div>
        </main>
      </div>
    </DefaultLayout>
  );
};

export default About;
