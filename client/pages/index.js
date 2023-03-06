import DefaultLayout from "../layouts/DefaultLayout";
import Image
 from "next/image";
import eye from '../static/eye.png'
import webcamIcon from '../static/webcam.png'
const Home = () => {
  return (
    <DefaultLayout>
      <div className="">
        <main className="">

          <header className=" grid-cols-2 w-max-[1600px] m-auto  ">



          <h2 className=" text-6xl font-bold ">      
            <div className="grid grid-temp-col-2">
            <span>
              Turn your webcam into <br/>
          a parking sensor. </span>   
         <Image className="inline-block" width={10} height={50} src={eye}></ Image>   
            </div>

        
           </h2>

           <h4> With a HD webcam, you can turns your webcam into a smart computer vision <br></br>
            parking mointor  </h4>

          </header>

        </main>
      </div>
    </DefaultLayout>
  );
};

export default Home;
