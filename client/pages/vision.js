import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import { load } from "@tensorflow-models/coco-ssd";

const visionPage = () => {

  const [webcamApproved, setWebCamApproved] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [totalFrames, setTotalFrames] = useState(0)

  if(true) 
  {
    return (<div className="flex outline outline-1 outline-stone-900">

    {/* <Toolbar webcamApprove={webcamApproved} setWebCamApprove={setWebCamApproved}></Toolbar> */}
      <div>
      <DrawingCanvas></DrawingCanvas>
      <ClientRender loaded={loaded} webcamApprove={webcamApproved} setTotalFrames={setTotalFrames} setLoaded={setLoaded}></ClientRender>
      </div>
      <RoisFeed totalFrames={totalFrames}></RoisFeed>

    </div>)
  }

};

export default visionPage;
