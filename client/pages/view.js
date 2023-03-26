

import { useRouter } from "next/router";
import { useEffect, useRef, useState } from "react";

import { totalOccupied } from "../libs/utillity";

const view = () => {

    const router = useRouter()
    const [counts, setCounts] = useState({OccupiedCount: "...", availableCount:"..."})
    const peerRef = useRef(null)

    const handleData = (data) => {
        console.log(data);
        console.log('1');
        let updatedCounts = totalOccupied(data);
        setCounts(updatedCounts)
    } 

    useEffect(() => 
    { 
    const initPeerJS = async () => {
        const { default: Peer } = await import("peerjs");
        const peer = new Peer();
        peer.on('open', (id) => 
        {
            let conn = peer.connect(router.query.remoteID);
            conn.on('data', (data) => 
            {
                handleData(data)
            })
        })
        peerRef.current = peer
      };
      initPeerJS()
    }, [])


    return ( 
        <> 
            <nav className="flex  border-b-2 border-black justify-around items-center h-[80px] bg-filler text-white"> 
                <h4 onClick={(e) => { router.push("https://www.sam-brink.com/")}} className="text-center uppercase  hover:cursor-pointer font-bold text-4xl"> Parker </h4>
                <span onClick={(e) => { router.push("https://www.sam-brink.com/about")}} className="text-center hover:cursor-pointer font-bold text-xl"> About </span>
             </nav>
    <div className="h-screen   flex flex-col  min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-filler w-full grow items-center">
         <div className="grid grid-rows-2 md:grid-rows-none md:grid-cols-2   w-full h-full">

      <div className="flex items-center w-full  justify-center flex-col gap-2 bg-green-700 ">

        <p className="font-bold text-6xl  rounded text-white">Available</p>
        <span className="text-5xl text-white ">{counts.availableCount}</span>
      </div>


      <div className="flex bg-red-700  justify-center w-full items-center flex-col gap-2">
        <p className="font-bold  text-6xl border-red-700 rounded text-white ">Occupied</p>
        <span className="  text-5xl text-white ">{counts.OccupiedCount}</span>
      </div>
    </div>

    </div> </>);
}
 
export default view;