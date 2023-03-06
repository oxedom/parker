const PreMenu = ({setWebcamEnable, setDemo}) => {

    const handleClick = (value) => {
        switch(value) 
        {
        case 'demo': {  setTimeout(() => { setDemo(true)}, 200) 
        break;
        }
        case 'webcam': {  setTimeout(() => { setWebcamEnable(true)}, 200) 
        break;
        }

        }
    }
    

    return ( <main className="text-white m-4 text-6xl flex gap-10 flex-col sm:flex-row">

        <button value={'demo'} onClick={(e) => { handleClick(e.target.value) }} className="bg-orange-600 p-5 rounded-lg hover:scale-105 duration-300" >Demo </button>
        <button  value={'webcam'}  onClick={(e) => { handleClick(e.target.value) }} className="bg-orange-600 p-5 rounded-lg hover:scale-105 duration-300">Use Webcam </button>
    </main>  );
}
 
export default PreMenu;