import ClientRender from "../components/modes/ClientRender";
import DrawingCanvas from "../components/DrawingCanvas";
import Toolbar from "../components/Toolbar";
import RoisFeed from "../components/RoisFeed";
import { useState } from "react";
import ToolbarTwo from "../components/ToolbarTwo";

const visionPage = () => {
  const [webcamApproved, setWebCamApproved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [totalFrames, setTotalFrames] = useState(0);

  if (true) {
    return (
      <div className="flex  p-16 outline outline-1  outline-stone-900">
        {/* <Toolbar webcamApprove={webcamApproved} setWebCamApprove={setWebCamApproved}></Toolbar> */}

        <div className="flex border-2 border-black">
          <RoisFeed totalFrames={totalFrames}></RoisFeed>

          <div className="">
            <DrawingCanvas></DrawingCanvas>
            <ClientRender
              loaded={loaded}
              webcamApprove={webcamApproved}
              setTotalFrames={setTotalFrames}
              setLoaded={setLoaded}
            ></ClientRender>
          </div>

          <div className="bg-green-500 border-l-2 border-black w-[250px]">
            <ToolbarTwo
              webcamApproved={webcamApproved}
              setWebCamApproved={setWebCamApproved}
            >
              {" "}
            </ToolbarTwo>
          </div>
        </div>
      </div>
    );
  }
};

export default visionPage;
