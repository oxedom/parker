import Image from "next/image";
import DefaultLayout from "../layouts/DefaultLayout";
import tensorflowIcon from "../public/static/tesnorflow.png";
import nextjsIcon from "../public/static/nextjs.png";
const About = () => {
  return (
    <DefaultLayout>
      <div className="flex-1 flex mx-auto flex-col p-10 max-w-[800px] gap-5 mt-5items-stretch  ">
        <main className="">
          <div className="text-lg  ">
            <h2 className="text-5xl text-center font-bold  rounded-lg  mb-10    text-orange-600">
              {" "}
              About parker
            </h2>
            <section className="p-5  rounded-md mt-2 flex gap-2 flex-col text-xl  ">
              <p className="">
                {" "}
                <span className="text-2xl text-gray-700 font-bold">
                  Parker{" "}
                </span>{" "}
                is a open source smart parking tool that to enable users to
                mointor parking spaces using their Webcam/CCTV cameras.
              </p>

              <p className="text-3xl mt-10 text-gray-700 ">Built with</p>

              <ul className="flex flex-col gap-2">
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

                <li>
                  {" "}
                  <a
                    className="text-blue-500"
                    href="https://github.com/WongKinYiu/yolov7"
                  >
                    {" "}
                    YOLO7{" "}
                  </a>{" "}
                </li>
                <li>
                  {" "}
                  <a
                    className="text-blue-500"
                    href="https://github.com/hugozanini/yolov7-tfjs"
                  >
                    {" "}
                    YOLO7-tfjs ( YOLO7 Model for TFJS Thanks Hugo!){" "}
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
