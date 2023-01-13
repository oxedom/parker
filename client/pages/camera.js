import Layout from "../layouts/DefaultLayout";
import { useCallback, useEffect, useRef, useState } from "react";
import React from "react";
import CanvasInput from "../components/CanvasInput";
import DrawingCanvas from "../components/DrawingCanvas";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
const Camera = () => {

  const outputRef = useRef(null);
  const [imageWidth, setImageWidth] = useState(1280);
  const [imageHeight, setImageHeight] = useState(720);
  const [fps, setFps] = useState(1000);
  const [track, setTrack] = useState(null);
  const [processing, setProcessing] = useState(false)
  const [selected, setSelected] = useState([])


  const getVideo = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { min: imageWidth } },
    });
    const track = stream.getVideoTracks()[0];
    setTrack(track);
  }, []);
  useEffect(() => {
    getVideo();
  }, []);

  return (
    <Layout>
        {!processing ?
        <div className="flex  justify-center m-2  ">
          <>
          <Toolbar></Toolbar>
          <div className="border-t-2 border-indigo-600" >

      
          <div className="cursor-crosshair pt-10">
        <DrawingCanvas imageWidth={imageWidth} imageHeight={imageHeight} setSelected={setSelected} selected={selected}>
          {" "}
        </DrawingCanvas>
        <CanvasInput
          track={track}
          fps={fps}
          outputRef={outputRef}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
        />
        </div>
        </div>
          <RoisFeed selected={selected}></RoisFeed>
          
          </>



        </div>
          :
          <>
               <img
          className=""
          width={imageWidth}
          height={imageHeight}
          ref={outputRef}
        />
          
          
          </>
   
        }
    </Layout>
  );
};

export default Camera;
