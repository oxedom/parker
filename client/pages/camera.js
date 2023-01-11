import Layout from "../layouts/DefaultLayout";
import Canvas from "../components/Canvas";
import Video from "../components/Video";
import { useEffect, useRef, useState } from "react";
import { setSize} from "../libs/canvas_utility"

const Camera = () => {
  // renderRectangle.printHello();
  const overlayRef = useRef(null)
  const canvasRef = useRef(null)
  const inputCanvasRef = useRef(null)
  const videoRef = useRef(null)
  const outputRef = useRef(null)
  
  const [imageWidth, setImageWidth] = useState(640)
  const [imageHeight, setImageHeight] = useState(480);
  const [stream, setStream] = useState(null)
  const arrOfScreens = [outputRef, videoRef, inputCanvasRef, canvasRef, overlayRef];


  useEffect(() => {
    const getVideo = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { min: imageWidth } },
      });
    
      //Gets the current screen
      const track = stream.getVideoTracks()[0];
    
      let { width, height } = track.getSettings();

    
      setInterval (() => {
        // intervalProcessing(track);
 
      }, 1000);
    };
    getVideo()
  }, [])

  return (
    <Layout>
      <div className="relative">
        <Canvas ref={overlayRef} imageWidth={imageWidth} imageHeight={imageHeight} />
        <Canvas ref={canvasRef} imageWidth={imageWidth} imageHeight={imageHeight} />
        <Canvas ref={inputCanvasRef} imageWidth={imageWidth} imageHeight={imageHeight} />
        <Video imageWidth={imageWidth} imageHeight={imageHeight} />
      </div>
      <img width={imageWidth} height={imageHeight} ref={outputRef}/>
    </Layout>
  );
};

export default Camera;
