import DefaultLayout from "../layouts/DefaultLayout";

const About = () => {
  return (
    <DefaultLayout>
      <div className="flex-1 flex mx-auto flex-col p-10 max-w-[1000px] gap-5 mt-20  items-stretch  ">
        <main className=" animate-fade">
          <div className="text-lg  ">
            <h2 className="text-5xl text-center font-bold pt-5 text-centerrounded-lg  text-white">
              {" "}
              About parker
            </h2>
            <section className="p-5 rounded-md mt-2 flex gap-2 flex-col text-xl bg-white  bg-opacity-75 border-orange-500 border-2">
              <p className="">
                {" "}
                <span className="text-2xl font-bold">Parker </span> is a open
                source smart parking tool that to enable users to mointor
                parking spaces using their Webcam/CCTV cameras.
              </p>

              <p>The technology's that run the app are the following: </p>
              <ul>
                <li> TensorFlow.js </li>
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
                    YOLO7-tfjs (Thanks Hugo!){" "}
                  </a>{" "}
                </li>
                <li> NextJS </li>
              </ul>

              <p>
                TensorFlow.js is a machine learning libary for Javascript that
                allows for fast client side ML directly in the browser.
              </p>
            </section>
          </div>
        </main>
      </div>
    </DefaultLayout>
  );
};

export default About;
