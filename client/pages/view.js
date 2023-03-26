
import { data } from "autoprefixer";
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
            let conn = peer.connect(router.query.remoteID);
            console.log(conn)
            conn.send('a')

            conn.on('data', (data) => 
            {
                console.log(data);
            })
        })

        // peer.on('connection', (conn) => 
        // {
        //     conn.on('open', () => {
        //         console.log('open on view')
        //     })

        //     conn.on('data', (d) => {
        //         console.log(data)
        //     })

        // })

      
        peerRef.current = peer
      };

      initPeerJS()
   
    })


    return ( <div className="h-screen gap-2 pt-10 flex flex-col   min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-filler w-full grow items-center">

    </div> );
}
 
export default view;