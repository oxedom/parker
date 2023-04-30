import { useEffect, useRef, useState } from "react";
import Head from "next/head";
import { useRouter } from "next/router";

const Input = () => {
  const inputRef = useRef(null);
  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const callRef = useRef(null);
  const router = useRouter();
  const [connection, setConnection] = useState(false);

  useEffect(() => {
    const initNoSleep = async () => {
      const { default: NoSleep } = await import("nosleep.js");
      const noSleep = new NoSleep();
      try {
        noSleep.enable();
      } catch (error) {
        console.log(error);
      }
    };

    const initPeerJS = async () => {
      const { default: Peer } = await import("peerjs");
      const peer = new Peer();
      peerRef.current = peer;
    };
    initNoSleep();
    initPeerJS();
  }, []);

  const shareVideo = async () => {
    try {
      const videostream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { exact: "environment" } },
      });
      streamRef.current = videostream;
      inputRef.current.srcObject = videostream;
      return videostream;
    } catch (error) {
      try {
        const videostream = await navigator.mediaDevices.getUserMedia({
          video: true,
          aspectRatio: { ideal: 16 / 9 },
        });

        streamRef.current = videostream;
        inputRef.current.srcObject = videostream;
        return videostream;
      } catch (error) {
        console.error(error);
      }
    }
  };

  const hangup = () => {
    if (callRef.current != null) {
      callRef.current.close();
      setConnection(false);
      streamRef.current = null;
      inputRef.current.srcObject = null;
    }
  };

  const call = async () => {
    const videoStream = await shareVideo();
    inputRef.current.srcObject = videoStream;
    const call = peerRef.current.call(router.query.remoteID, videoStream);
    callRef.current = call;
    call.on("stream", async (remoteStream) => {
      setConnection(true);
    });
  };

  return (
    <>
      <Head>
        <title> Parkerr: Input </title>
      </Head>

      <div className="h-screen gap-2 pt-10 flex flex-col   min-h-screen  bg-fixed bg-no-repeat bg-cover  bg-filler w-full grow items-center">
        <p className="text-5xl py-2 text-white ">
          {" "}
          Connection: {connection ? "Established" : "Pending"}{" "}
        </p>

        <div className="flex flex-col md:flex-row gap-4">
          <button
            className="bg-green-400 py-2 rounded-lg shadow-sm active:bg-green-600 hover:bg-green-500 text-white  font-bold text-4xl p-5 w-[250px]"
            onClick={call}
          >
            {" "}
            Call
          </button>
          <button
            className="bg-red-400 py-2 rounded-lg shadow-sm active:bg-red-600 hover:bg-red-500 text-white  font-bold text-4xl p-5 w-[250px]"
            onClick={hangup}
          >
            {" "}
            Hang up
          </button>
          <video
            autoPlay={true}
            className=" rounded-xl "
            ref={inputRef}
          ></video>
        </div>
      </div>
    </>
  );
};

export default Input;
