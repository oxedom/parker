import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import { load } from "@tensorflow-models/coco-ssd";

const visionPage = () => {

  const [webcamApprove, setWebCamApprove] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [totalFrames, setTotalFrames] = useState(0)

  if(true) 
  {
    return (<div className="flex m-2 gap-2">

    <Toolbar webcamApprove={webcamApprove} setWebCamApprove={setWebCamApprove}></Toolbar>
      <div>
      <DrawingCanvas></DrawingCanvas>
      <ClientRender webcamApprove={webcamApprove} setTotalFrames={setTotalFrames} setLoaded={setLoaded}></ClientRender>
      </div>
      <RoisFeed totalFrames={totalFrames}></RoisFeed>

    </div>)
  }
  else 
  {
    return (<h1> Loading...</h1>)
  }
};

export default visionPage;
