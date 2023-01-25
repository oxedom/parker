

import { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
const cocoSsd = require("@tensorflow-models/coco-ssd");

const FaceTest = () => {

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(undefined);
  const [isLoaded, setIsLoaded] = useState(false);
  const [webcamEnabled, setWebcamEnabled] = useState(false);



  const handleLoadWaiting = async () => {


    return new Promise((resolve) => {

      const timer = setInterval(() => {
        const conditions = (
          webcamRef.current &&
          webcamRef.current.video &&
          webcamRef.current.video.readyState === 4
        );
        if (conditions) {
          resolve(true);
          clearInterval(timer);
        }
      }, 500);
    });
  };



  const renderWebcam = () => (
    <>
      <Webcam audio={false} ref={webcamRef} className={!isLoaded ? 'video' : 'video frame'} />
      <canvas ref={canvasRef} className='video' />
    </>
  )

  useEffect(() => {

    cocoSsd.load().then(loadedModel => {
      setModel(loadedModel);
  });
  


    if (webcamEnabled) {
      setIsLoaded(false);
      objectDecterHandler();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [webcamEnabled]);

  function predictWebcam() {
    // Now let's start classifying the stream.
    model.detect(webcamRef.current.video).then(function (predictions) {
      // Remove any highlighting we did previous frame.
      // Now lets loop through predictions and draw them to the live view if
      // they have a high confidence score.
      for (let n = 0; n < predictions.length; n++) {
        console.log(predictions);
        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) {
          const p = document.createElement("p");
          p.innerText =
            predictions[n].class +
            " - with " +
            Math.round(parseFloat(predictions[n].score) * 100) +
            "% confidence.";
          // Draw in top left of bounding box outline.
          p.style =
            "left: " +
            predictions[n].bbox[0] +
            "px;" +
            "top: " +
            predictions[n].bbox[1] +
            "px;" +
            "width: " +
            (predictions[n].bbox[2] - 10) +
            "px;";

          // Draw the actual bounding box.
          const highlighter = document.createElement("div");
          highlighter.setAttribute("class", "highlighter");
          const left_x = predictions[n].bbox[0];
          const top_y = predictions[n].bbox[1];
          const width = predictions[n].bbox[2];
          const height = predictions[n].bbox[3];
          console.log(predictions);
          const obj = { left_x, top_y, width, height };
          console.log(obj);

     
        }
      }

      // Call this function again to keep predicting when the browser is ready.
      window.requestAnimationFrame(predictWebcam);
    });
  }




  const objectDecterHandler = async () => 
  {
    console.log("Object decter handler");
    await handleLoadWaiting();
    console.log("Loading done");
    console.log(webcamRef.current.video);
    if(webcamRef.current.video) 
    {
      setIsLoaded(true)
      console.log("Webcam loaded");
      predictWebcam()

    }

  }

  return (
    <div className='container'>
      <header className='header'>
        <h1>EmojiMask</h1>
        <Webcam ref={webcamRef} ></Webcam>
        <label className='switch'>
          <input type='checkbox' checked={webcamEnabled} onClick={() => setWebcamEnabled((previous) => !previous)} />
          <span className='slider' />
        </label>
      </header>

      {!isLoaded && webcamEnabled}

      <main className='main'>
        {webcamEnabled && renderWebcam()}
      </main>


 
    </div>
  );
}

export default FaceTest;