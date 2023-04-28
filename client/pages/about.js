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
import Question from "../components/Question";

const About = () => {

  const questions  = 
  [
    { 
      question: "How does it Parkerr work?" , 
      answer: "Under the hood, Parkerr uses tensorflow.js a javascript library to deploy machine learning models entirely in the browser. Parkerr is running YOLO7 (You Only Look Once v7) a real-time object detection computer vision model. YOLO is one of the fastest, most accurate models and simplest to train models available publicly to the world. My implementation of Parkerr involves mapping the model's output to a custom data structure and integrating it into the application's data pipeline and state.",
    },
    { 
      question: "How does remote communication work?" , 
      answer: "Parkerr currently implements remote communication through a P2P solution using PeerJS, a javascript library that implements correct WebRTC configuration using PeerServer Cloud services and Google stuns servers.",
    },
    { 
      question: "What kind of video input does the application take?" , 
      answer: "Parkerr can take webcam footage by allowing access through your browser or remote video communication through another device that has a built-in camera and modern browser.",
    },
    { 
      question: "What data is collected from the application?" , 
      answer: "Other than site website analytics, none. Parkerr processes all of the video footage on the client side and all remote communication is p2p and private.",
    },
    { 
      question: "How was the auto detect parking implemented?" , 
      answer: "The auto-detect feature works by collecting a snapshot of all of the detected vehicles for a few seconds and then suppresses all of the vehicles that have not been in a consistent position. If you were to try and use the auto-detect feature on an empty parking lot no selections will be made, if attempted on a full parking lot then all the parking spaces will be marked.",
    },
    { 
      question: "Could the procesing be moved to a backend server?" , 
      answer: "Yes! That is possible and was implemented, Originally the architecture of this project, I created a flask server that receives base64/blobs from the client and processes the input using yolo3 model with OpenCV, works fine on a small scale but needs additional configuration and refactoring for larger scales. Such as a server with a powerful GPU and autoscaling.",
    },
    { 
      question: "What objects can the model detect?" , 
      answer: "human, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.",
    },
  ]

  return (
    <DefaultLayout>
      <Head>
      <title> Parkerr: About</title>
      </Head>
      <div className="flex-1 flex mx-auto flex-col p-10 max-w-[800px] gap-5 mt-5items-stretch  ">
        <main className="">
          <div className="text-lg  ">
            <h2 className="text-5xl text-center flex  justify-center items-center gap-5 font-bold  rounded-lg  mb-10    text-orange-600">
              {" "}
              About Parkerr
              <div>

                <a href="https://github.com/oxedom/parker" className="  ">               <Image width={50} src={github}></Image>    </a>
              </div>
            </h2>
            
            <section className="p-5  rounded-md flex gap-2 flex-col text-xl  ">
         

              <p className="b-t border-black">
                {" "}
                <span className="text-2xl text-gray-700 font-bold">
                  Parkerr{" "}
                </span>{" "}
                is browser application that allows you to detect and monitor parking spaces to using computer vision.
                
              </p>


              <p className="text-4xl my-5   text-gray-700 ">Built with</p>

              <ul className="grid grid-cols-none  sm:grid-cols-2 gap-2">
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
                  {/* <Image width={50} alt="yolo" src={yoloLogo}></Image> */}
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                     YOLO7-tfjs 
                  </a>{" "}
                  
                </li>
    
                <li className="flex gap-2 text-center items-center">
                  {" "}
                  <Image width={30} alt="tailwind" src={tailwindLogo}></Image>
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
                  {/* <Image width={60} alt="recoil" src={recoilLogo}></Image> */}
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                    React Recoil 
                  </a>{" "}
                  
                </li>
         
              </ul>
            </section>

          <section>
          <h3 className="font-bold border-b border-gray-700 text-center text-3xl  p-2"> F.A.Q  </h3>
           

            {questions.map(q => <Question key={q.question} answer={q.answer}  question={q.question}> </Question>)}
   


          </section>


            <section className="flex flex-col gap-5">
              <h3 className="font-bold border-b border-gray-700 text-center text-3xl text-gray-800"> Instructions  </h3>
                
              <div className="mt-1 ">
            <h4 className="text-xl font-bold text-center ">
              <a className="text-blue-500 text-center border-b p-1 border-black " href="https://github.com/oxedom/parker"> More information in the repo</a>
            </h4>
            </div>

            <div className="mt-1 ">
            <h4 className="text-2xl font-bold ">
            Mobile Phone Camera Instructions
            </h4>
            <p>
            After entering the vision page and pressing the remote stream button, you can scan QR code with your mobile device and when loaded,press on the call button to allow acesss to your phones camera. <br></br>
            <br></br>
            <div className="">
              <h4> </h4>
              <p>     1. Make sure phone is streaming video in landscape mode</p>
              <p>
              2. Change your settings that your phone screen doesn't autolock to enable a continutes video stream. 
              </p>
              <p> 3.You might need to press the call button a few times to establish the connection 
              </p>
            </div>
     
       

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


            
   

            <div className="mb-10 mt-2 ">
            <h4 className="text-2xl font-bold ">
           Special Thanks
            </h4>
            <ul className="flex flex-col gap-2">
              <li>Hugo zanini, for porting YOLO7 to tfjs and for sharing his code</li>
              <li> Jason Mayes, for  advice and guidance </li>
              <li> The tensorflow Community! </li>
              <li> Everyone who believed in me 😊 🚗  </li>
            </ul>
            </div>
            </section>

          </div>
        </main>
      </div>
    </DefaultLayout>
  );
};

export default About;
