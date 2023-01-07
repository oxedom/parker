import Layout from "../layouts/DefaultLayout";
import Webcam from "../components/Webcam";
import Image from 'next/image'
import { renderRectangleFactory } from "../libs";

import { useEffect, useRef } from "react";

const Camera = () => {
  // renderRectangle.printHello();
const overlayRef = useRef(null);
const canvasRef = useRef(null);
const outputRef = useRef(null);
const inputCanvasRef = useRef(null);
const webcamRef = useRef(null)

useEffect(() => {
  const renderRectangle = renderRectangleFactory(
    canvasRef.current,
    overlayRef.current
  );
}, []);


  return (
    <Layout>
      <div className="relative">
        <h4>Welcome to Camera page</h4>
        <canvas className="absolute" ref={overlayRef}></canvas>
         <canvas className="absolute" ref={canvasRef}></canvas>
         <canvas className="absolute z-[-99]"  ref={inputCanvasRef}
      ></canvas>


         <video
         className="invisible"
        ref={webcamRef}
        // style="visibility: hidden"
      ></video>

        <Webcam outputRef={outputRef.current} overlayRef={overlayRef.current} canvasRef={canvasRef.current} inputCanvasRef={inputCanvasRef.current}></Webcam>
      </div>
    </Layout>
  );
};

export default Camera;
