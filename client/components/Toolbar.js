const Toolbar = ({}) => {
    return ( <div className="border-2 border-indigo-600 w-80 flex justify-center rounded-l-lg">
    <div className="bg-gray-500 ">

    <h2 className="bg-white border-2 border-black "> Toolbar </h2>
    <p className=""> Select Square Color</p>
   <input type='number' className="border-2 border-black"  placeholder="FPS"/>
    <div className="p-2 hover:bg-green-200 bg-red-500 cursor-pointer" > Start</div>
    </div>

    </div>  );
}
 
export default Toolbar;