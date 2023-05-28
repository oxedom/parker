import { useRouter } from "next/router";
import { useEffect, useRef, useState } from "react";

import { totalOccupied } from "../libs/utillity";

export default function View() {
  const router = useRouter();
  const [counts, setCounts] = useState({
    OccupiedCount: "...",
    availableCount: "...",
  });
  const peerRef = useRef(null);

  const handleData = (data) => {
    let updatedCounts = totalOccupied(data);
    setCounts(updatedCounts);
  };

  useEffect(() => {
    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();
      peer.on("open", (id) => {
        let conn = peer.connect(router.query.remoteID);
        conn.on("data", (data) => {
          handleData(data);
        });
      });
      peerRef.current = peer;
    };
    initPeerJS();
  }, []);

  return (
    <>
      <nav className="flex  border-b-2 border-black justify-around items-center h-[80px] bg-filler text-white">
        <h4
          onClick={(e) => {
            router.push("https://www.parkerr.org/");
          }}
          className="text-4xl font-bold text-center uppercase hover:cursor-pointer"
        >
          Parkerr
        </h4>

        <span
          onClick={(e) => {
            router.push("https://www.parkerr.org/about");
          }}
          className="text-xl font-bold text-center hover:cursor-pointer"
        >
          About
        </span>
      </nav>
      <div className="flex flex-col items-center w-full h-screen min-h-screen bg-fixed bg-no-repeat bg-cover bg-filler grow">
        <div className="grid w-full h-full grid-rows-2 md:grid-rows-none md:grid-cols-2">
          <div className="flex flex-col items-center justify-center w-full gap-2 bg-green-700 ">
            <p className="text-6xl font-bold text-white rounded">Available</p>
            <span className="text-5xl text-white ">
              {counts.availableCount}
            </span>
          </div>

          <div className="flex flex-col items-center justify-center w-full gap-2 bg-red-700">
            <p className="text-6xl font-bold text-white border-red-700 rounded ">
              Occupied
            </p>
            <span className="text-5xl text-white ">{counts.OccupiedCount}</span>
          </div>
        </div>
      </div>
    </>
  );
}
