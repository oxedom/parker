const AppNavbar = ({
    openModal}) => {

    let btnClass = 
    "bg-gray-200 p-2 m-1 shadow-neooutline-[1.5px] border border-black text-bold  rounded-2xl shadow-neo  hover:shadow-neo-hover shadow-outline"


    return ( <nav className="bg-blue-300 shadow-neo p-5 border-black border mb-4 grid gap-2 grid-cols-2 "> 
    <div></div>
    <div className=" grid gap-2 grid-cols-3 grid-flow-row
    ">

    <button onClick={openModal} className={`${btnClass} order-last` }> Settings </button>
    <div></div>
    <div></div>
    </div>

    </nav> );
}
 
export default AppNavbar;