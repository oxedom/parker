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

          <header className="flex flex-col items-center self-center justify-center gap-2 w-max-[800px] ">


          <h2 className=" text-6xl font-bold flex-col  gap-2 ">      
          {/* Empowering your webcam to do more */}
          <h1>
            <div className="flex">
            <span>Empowering your webcam </span>   
         <Image className="inline-block" width={75} height={50} src={webcamIcon}></ Image>   
            </div>

              <div className="flex justify-center">
              <span > to do more
          <Image className="ml-55 inline-block" width={100} height={50} src={eye}></ Image>     
             </span>
              </div>
 

          </h1>


   
        
          

    

          
          
           </h2>
          </header>

        </main>
      </div>
    </DefaultLayout>
  );
};

export default Home;
