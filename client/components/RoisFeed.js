const RoisFeed = ({selected}) => {

    return ( <div className="border-2 border-indigo-600 w-80  rounded-r-lg flex flex-col"> 
    <h4 className="text-3xl border-2 border-black"> Selected ROI's</h4>
    {selected.map(s => <h1 key={s.uid}> {s.cords.right_x}</h1>) }
    </div> );
}
 
export default RoisFeed;