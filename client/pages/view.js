
import { useRouter } from "next/router";
import { useEffect, useRef } from "react";

const view = () => {

    const router = useRouter()
    const peerRef = useRef(null)
    useEffect(() => 

    {
            
    const initPeerJS = async () => {
      
        const { default: Peer } = await import("peerjs");
        const peer = new Peer();

        peer.on('open', (id) => 
        {
        var conn = peer.connect(router.query.remoteID);
        conn.on('data', (d) => 
        {
            console.log(d);
        })

        })


      
        peerRef.current = peer
      };

      initPeerJS()
   
    })


    return ( <div className="h-screen gap-2 pt-10 flex flex-col   min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-filler w-full grow items-center">

    </div> );
}
 
export default view;