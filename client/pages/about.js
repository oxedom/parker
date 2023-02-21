import DefaultLayout from "../layouts/DefaultLayout";

const About = () => {
    return (
        
        <DefaultLayout>
        <div className="flex-1 flex mx-auto flex-col p-10 max-w-[1000px] gap-5 mt-20  items-stretch  ">
        
        <main className="shadow-neo-sm " >

      
        <div className="bg-slate-100 p-5 text-lg  " > 
        <h2 className="text-5xl font-bold pt-5 text-center bg-purple-400 p-5 rounded-lg  text-white"> About parker...</h2>
        <p> Hello Everyone and Welcome to parker.com, Parker is a first of it's kind Web app that uses tensorflowJS</p>
        </div>
        </main>


    </div>
    </DefaultLayout>
     );
}
 
export default About;